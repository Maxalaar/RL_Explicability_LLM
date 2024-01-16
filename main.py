from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions
from text_cart_pole import TextCartPole
from trainer import Trainer

if __name__ == '__main__':
    environment: TextCartPole = TextCartPole()
    
    model_generates_instructions = ModelGeneratesInstructions(
        model_id='gpt2-large',  # 'gpt2-large' or 'gpt2'
        instruction_size_max=50,
        load_model_reference_in_4bit=True,
        number_layers_freeze=34,     # 32
    )

    model_follows_instructions = ModelFollowsInstructions(
        model_id='mistralai/Mistral-7B-v0.1',   # 'mistralai/Mistral-7B-v0.1' or 'gpt2'
        load_in_4bit=True,
        list_actions_tokens=environment.get_actions_tokens(),
        size_response_action=30,
        use_logit_to_predict=True,
    )

    trainer: Trainer = Trainer(model_generates_instructions, model_follows_instructions, environment, batch_size=30)
    trainer.learn(1_000_000_000)

