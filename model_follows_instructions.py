from typing import List, Union
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import regex
import torch


class ModelFollowsInstructions:
    def __init__(self, model_id: str, list_actions_tokens: List[str], size_response_action: int = None, load_in_8bit=False, load_in_4bit=False, use_logit_to_predict=False):
        self.model_id = model_id
        self.use_logit_to_predict = use_logit_to_predict
        self.device: str = 'auto'
        self.size_response_action = size_response_action

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, quantization_config=bnb_config, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.list_actions_tokens: List[str] = list_actions_tokens
        self.list_actions_tokens_ids = [self.tokenizer.encode(token, return_tensors='pt') for token in self.list_actions_tokens]
        self.actions_regex_pattern = '|'.join(map(regex.escape, self.list_actions_tokens))

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors='pt')

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def follow_instructions(self, instructions: str, observation: str) -> Union[str, None]:
        prompt = 'Follow these instructions : ' + instructions + '\n' + observation + '\nThe action based on the previous instructions is :'
        input_ids = self.encode(prompt)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        if self.use_logit_to_predict:
            return self.action_logit(input_ids, attention_mask)
        else:
            return self.action_text(input_ids, attention_mask, prompt)

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()
    
    def action_text(self, input_ids, attention_mask, prompt):
        response_text = self.decode(self.model.generate(input_ids.to('cuda'), max_new_tokens=self.size_response_action, attention_mask=attention_mask, pad_token_id=50256)[0])     # , eos_token_id=self.eos_token_id
        response_text = response_text.replace(prompt, '')
        return self.find_action_token(response_text)
    
    def find_action_token(self, text: str) -> Union[str, None]:
        match = regex.search(self.actions_regex_pattern, text)

        if match:
            return match.group()
        else:
            return None
    
    def action_logit(self, input_ids, attention_mask):
        score = []
        outputs = self.model.generate(input_ids.to('cuda'), pad_token_id=50256, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
        for action_token_ids in self.list_actions_tokens_ids:
            wanted_seq = torch.cat((input_ids, action_token_ids), dim=1).to('cuda')
            score.append(self.model.compute_transition_scores(wanted_seq, outputs.scores, normalize_logits=False))

        return self.list_actions_tokens[torch.argmax(torch.tensor(score)).item()]




