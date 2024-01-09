from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer
from trl.core import respond_to_batch


class ModelGeneratesInstructions:
    def __init__(self, model_id: str, instruction_size_max: int = None, instructions_prompt: str = 'Provide a set of instructions to solve the following task: '):
        self.model_id: str = model_id
        self.device = 'cuda'
        self.instruction_size_max: int = instruction_size_max

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id, device_map=self.device)
        self.model_reference = create_reference_model(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.instructions_prompt = instructions_prompt

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors='pt').to(self.device)

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def instruction_generation(self, description_environment: str) -> str:
        query_tensor = self.encode(self.instructions_prompt + description_environment)

        # Need to optimize
        optional_arguments = {}
        if self.instruction_size_max is not None:
            optional_arguments['txt_len'] = self.instruction_size_max

        response_tensor = respond_to_batch(self.model, query_tensor, **optional_arguments)
        return self.decode(response_tensor[0])

    def get_memory_footprint(self):
        pass
        # return self.model.get_memory_footprint()



