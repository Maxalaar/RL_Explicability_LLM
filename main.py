from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions
from text_cart_pole import TextCartPole
from trainer import Trainer

if __name__ == '__main__':
    environment: TextCartPole = TextCartPole()
    
    model_generates_instructions = ModelGeneratesInstructions(
        model_id='gpt2',  # 'gpt2-large', 'gpt2', 'stabilityai/stable-code-3b', 'openlm-research/open_llama_3b_v2'
        instruction_size_max=50,
        load_model_in_4bit=True,
        load_model_reference_in_4bit=True,
        use_lora=True,
    )

    model_follows_instructions = ModelFollowsInstructions(
        model_id='gpt2',   # 'mistralai/Mistral-7B-v0.1', 'gpt2'
        list_actions_tokens=environment.get_actions_tokens(),
        load_in_4bit=True,
        size_response_action=10,
        use_logit_to_predict=True,
    )

    trainer: Trainer = Trainer(model_generates_instructions, model_follows_instructions, environment, batch_size=50)
    trainer.learn(1_000_000_000)

