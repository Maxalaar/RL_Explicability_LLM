from environments.minimal_text import MinimalText
from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions
from environments.text_cart_pole import TextCartPole
from trainer import Trainer

if __name__ == '__main__':
    # environment: TextCartPole = TextCartPole()
    environment = MinimalText()
    
    model_generates_instructions = ModelGeneratesInstructions(
        model_id='gpt2',  # 'gpt2-large', 'gpt2', 'stabilityai/stable-code-3b', 'openlm-research/open_llama_3b_v2'
        instruction_size_max=50,
        use_lora=True,
        load_model_in_4bit=True,
        load_model_reference_in_4bit=True,
    )

    model_follows_instructions = ModelFollowsInstructions(
        model_id='mistralai/Mistral-7B-v0.1',   # 'mistralai/Mistral-7B-v0.1', 'gpt2'
        list_actions_tokens=environment.get_actions_tokens(),
        load_in_4bit=True,
        use_logit_to_predict=True,
        size_response_action=10,
    )

    trainer: Trainer = Trainer(model_generates_instructions, model_follows_instructions, environment, batch_size=10)
    trainer.learn(1_000_000_000)

