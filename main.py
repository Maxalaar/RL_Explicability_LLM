from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions
from text_cart_pole import TextCartPole
from trainer import Trainer

if __name__ == '__main__':
    environment: TextCartPole = TextCartPole()

    model_generates_instructions = ModelGeneratesInstructions('gpt2')
    model_follows_instructions = ModelFollowsInstructions('TheBloke/Mistral-7B-v0.1-AWQ', environment.get_actions_tokens())     # 'gpt2', 'TheBloke/Mistral-7B-v0.1-AWQ'

    trainer: Trainer = Trainer(model_generates_instructions, model_follows_instructions, environment, batch_size=3)
    trainer.learn(1_000_000_000)

