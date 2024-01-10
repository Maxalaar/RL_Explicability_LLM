from typing import List, Union
# from awq import AutoAWQForCausalLM
# from ctransformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl.core import respond_to_batch
import regex
import sys
import torch


class ModelFollowsInstructions:
    def __init__(self, model_id: str, list_actions_tokens: List[str], size_response_action: int = None, device: str = 'cuda:0', load_in_8bit=False, load_in_4bit=False):
        self.model_id = model_id
        self.device: str = device
        self.size_response_action = size_response_action

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, quantization_config=bnb_config, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # self.pad_token_id = self.tokenizer.pad_token_id
        # self.eos_token_id = self.tokenizer.eos_token_id

        # self.model.config.pad_token_id = self.pad_token_id
        # self.model.config.eos_token_id = self.eos_token_id

        # self.model = AutoAWQForCausalLM.from_quantized(self.model_id, fuse_layers=True, trust_remote_code=False, safetensors=True)  # device_map='sequential'
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)

        # self.llm = AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', model_file='mistral-7b-instruct-v0.1.Q4_K_M.gguf', model_type='mistral', gpu_layers=100_000_000, max_new_tokens=20, device_map=self.device)


        self.list_actions_tokens: List[str] = list_actions_tokens
        self.actions_regex_pattern = '|'.join(map(regex.escape, self.list_actions_tokens))

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors='pt').to(self.device)

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def follow_instructions(self, instructions: str, observation: str) -> Union[str, None]:
        # response_text = self.llm('\nFollow these instructions : ' + instructions + '\n' + observation + '\nThe action based on the previous instructions is :')
        
        # query_tensor = self.encode(instructions + observation)

        # # Need to optimize
        # optional_arguments = {}
        # if self.size_response_action is not None:
        #     optional_arguments['txt_len'] = self.size_response_action

        # response_tensor = respond_to_batch(self.model, query_tensor, **optional_arguments)
        # response_text = self.decode(response_tensor[0])
        
        prompt = '\nFollow these instructions : ' + instructions + '\n' + observation + '\nThe action based on the previous instructions is :'
        input_ids = self.encode(prompt)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        response_text = self.decode(self.model.generate(input_ids, max_new_tokens=self.size_response_action, attention_mask=attention_mask, pad_token_id=50256)[0])     # , eos_token_id=self.eos_token_id
        response_text = response_text.replace(prompt, '')


        return self.find_action_token(response_text)

    def find_action_token(self, text: str) -> Union[str, None]:
        # Need to optimize
        match = regex.search(self.actions_regex_pattern, text)

        if match:
            return match.group()
        else:
            return None

    def get_memory_footprint(self):
        pass
        # return self.model.get_memory_footprint()