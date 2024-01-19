from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer
from trl.core import respond_to_batch
from peft import LoraConfig


class ModelGeneratesInstructions:
    def __init__(
        self,
        model_id: str,
        instruction_size_max: int = None,
        instructions_prompt: str = 'Provide a set of instructions to solve the following task: ',
        load_model_reference_in_4bit: bool = False,
        load_model_reference_in_8bit: bool = False,
        use_lora: bool = False,
        load_model_in_4bit=False,
        load_model_in_8bit=False,
    ):
        self.model_id: str = model_id
        self.device: str = 'auto'
        self.instruction_size_max: int = instruction_size_max

        if use_lora:
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            self.lora_config = None

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_id,
            device_map=self.device,
            peft_config=self.lora_config,
            load_in_4bit=load_model_in_4bit,
            load_in_8bit=load_model_in_8bit,
            trust_remote_code=True,
        )

        self.model_reference = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_id,
            device_map=self.device,
            load_in_4bit=load_model_reference_in_4bit,
            load_in_8bit=load_model_reference_in_8bit,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.instructions_prompt = instructions_prompt

    def encode(self, query_text: str):
        return self.tokenizer.encode(query_text, return_tensors='pt')

    def decode(self, query_tensor) -> str:
        return self.tokenizer.decode(query_tensor, skip_special_tokens=True)

    def get_memory_footprint(self):
        # return self.model.get_memory_footprint()
        pass

    def freeze(self, number_layers_freeze):
        current_index_layer: int = 0
        for name, parameter in self.model.named_parameters():
            if current_index_layer < 8 * number_layers_freeze + 2:
                parameter.requires_grad = False
            current_index_layer += 1



