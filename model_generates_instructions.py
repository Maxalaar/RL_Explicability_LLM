from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer
from trl.core import respond_to_batch


class ModelGeneratesInstructions:
    def __init__(self, model_id: str, instruction_size_max: int = None, instructions_prompt: str = 'Provide a set of instructions to solve the following task: ', load_model_reference_in_4bit: bool = False, load_model_reference_in_8bit: bool = False, number_layers_freeze: int = 0):
        self.model_id: str = model_id
        self.device: str = 'auto'
        self.instruction_size_max: int = instruction_size_max

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id, device_map=self.device)

        self.freeze(number_layers_freeze)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.model_reference = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id, device_map=self.device, load_in_4bit=load_model_reference_in_4bit, load_in_8bit=load_model_reference_in_8bit)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.instructions_prompt = instructions_prompt

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors='pt')

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def instruction_generation(self, description_environment: str) -> str:
        query_tensor = self.encode(self.instructions_prompt + description_environment)

        optional_arguments = {}
        if self.instruction_size_max is not None:
            optional_arguments['txt_len'] = self.instruction_size_max

        response_tensor = respond_to_batch(self.model, query_tensor.to(self.model.current_device), **optional_arguments)
        return self.decode(response_tensor[0])

    def get_memory_footprint(self):
        # return self.model.get_memory_footprint()
        pass

    def freeze(self, number_layers_freeze):
        current_index_layer: int = 0
        for name, parameter in self.model.named_parameters():
            if current_index_layer < 8 * number_layers_freeze + 2:
                parameter.requires_grad = False
            current_index_layer += 1



