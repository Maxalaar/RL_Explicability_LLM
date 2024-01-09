from typing import List, Union
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.core import respond_to_batch
import regex


class ModelFollowsInstructions:
    def __init__(self, model_id: str, list_actions_tokens: List[str], size_response_action: int = None, load_in_8bit=False):
        self.model_id = model_id
        self.device = 'cuda'
        self.size_response_action = size_response_action

        # self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, load_in_8bit=load_in_8bit)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoAWQForCausalLM.from_quantized(self.model_id, fuse_layers=True, trust_remote_code=False, safetensors=True, device_map='sequential')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)

        self.list_actions_tokens: List[str] = list_actions_tokens
        self.actions_regex_pattern = '|'.join(map(regex.escape, self.list_actions_tokens))

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors="pt").to(self.device)

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def follow_instructions(self, instructions: str, observation: str) -> Union[str, None]:
        query_tensor = self.encode(instructions + observation)

        # Need to optimize
        optional_arguments = {}
        if self.size_response_action is not None:
            optional_arguments['txt_len'] = self.size_response_action

        response_tensor = respond_to_batch(self.model, query_tensor, **optional_arguments)
        response_text = self.decode(response_tensor[0])
        return self.find_action_token(response_text)

    def find_action_token(self, text: str) -> Union[str, None]:
        # Need to optimize
        match = regex.search(self.actions_regex_pattern, text)

        if match:
            return match.group()
        else:
            return None

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()