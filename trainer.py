import time
import random
from datetime import datetime
import torch
import numpy as np
import gymnasium
from trl import PPOTrainer, PPOConfig
from tensorboardX import SummaryWriter

from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions


class Trainer:
    def __init__(self, model_generates_instructions: ModelGeneratesInstructions, model_follows_instructions: ModelFollowsInstructions, environment, batch_size: int = 1):
        self.model_generates_instructions: ModelGeneratesInstructions = model_generates_instructions
        self.model_follows_instructions: ModelFollowsInstructions = model_follows_instructions

        self.environment: gymnasium.Env = environment

        self.batch_size: int = batch_size
        self.configuration = PPOConfig(batch_size=self.batch_size)
        self.generation_kwargs = {
            'min_length': -1,
            'top_k': 0.0,
            'top_p': 1.0,
            'do_sample': True,
            'pad_token_id': self.model_generates_instructions.tokenizer.eos_token_id,
            'max_new_tokens': self.model_generates_instructions.instruction_size_max,
            'return_prompt': False,
        }

        self.trainer = PPOTrainer(self.configuration, self.model_generates_instructions.model, self.model_generates_instructions.model_reference, self.model_generates_instructions.tokenizer)

        self.login_directory = './login/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(logdir=self.login_directory)

        self.print_initialisation_information()

        self.queries_tensor = self.queries_tensor_generations()

        self.instructions_generations_time = None
        self.learn_time = None
        self.follow_instruction_time = None
        self.episode_time = None
        self.batch_time = None

    def episode(self, instruction: str):
        total_reward: float = 0.0
        observation, info = self.environment.reset()
        terminated = False

        while not terminated:
            start_follow_instructions_time = time.time()
            action = self.model_follows_instructions.follow_instructions(instruction, observation)
            end_follow_instructions_time = time.time()
            self.follow_instruction_time.append(end_follow_instructions_time - start_follow_instructions_time)

            observation, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward

        self.environment.close()

        return total_reward

    def learn(self, number_steps: int):
        for i in range(number_steps):
            rewards = []
            
            self.instructions_generations_time = []
            self.learn_time = []
            self.follow_instruction_time = []
            self.episode_time = []
            self.batch_time = []

            start_batch_time = time.time()
            start_instructions_generations_time = time.time()
            responses_ids, responses_text = self.instructions_generations()
            end_instructions_generations_time = time.time()
            self.instructions_generations_time.append(end_instructions_generations_time - start_instructions_generations_time)

            for j in range(self.batch_size):
                start_step_time = time.time()
                rewards.append(torch.tensor(self.episode(responses_text[j])))
                end_step_time = time.time()
                self.episode_time.append(end_step_time - start_step_time)

            start_learn_time = time.time()
            self.trainer.step(self.queries_tensor, responses_ids, rewards)
            end_learn_time = time.time()
            end_batch_time = time.time()
            self.learn_time.append(end_learn_time - start_learn_time)
            self.batch_time.append(end_batch_time - start_batch_time)

            self.print_step_information(i, responses_ids, rewards)

    def queries_tensor_generations(self):
        queries_tensor = []
        query = self.model_generates_instructions.encode(self.environment.description_environment)
        for _ in range(self.batch_size):
            queries_tensor.append(query[0])
        return queries_tensor

    def instructions_generations(self):
        ids_instructions = self.trainer.generate(
            query_tensor=self.queries_tensor,
            **self.generation_kwargs,
        )
        text_instructions = []
        for instruction in ids_instructions:
            text_instructions.append(self.model_generates_instructions.decode(instruction))
        pass

        return ids_instructions, text_instructions

    def print_initialisation_information(self):
        print('The model for generates instructions:')
        print('id: ' + self.model_generates_instructions.model_id)
        print('device: ' + self.model_generates_instructions.device)
        print('memory: ' + str(self.model_generates_instructions.get_memory_footprint()))
        print('')
        print('The model for follows instructions:')
        print('id: ' + self.model_follows_instructions.model_id)
        print('device: ' + self.model_follows_instructions.device)
        print('memory: ' + str(self.model_follows_instructions.get_memory_footprint()))
        print('')
        print('Learning information:')
        print('batch size: ' + str(self.batch_size))
        print('instructions prompt: "' + self.model_generates_instructions.instructions_prompt + self.environment.description_environment + '"')
        print('observation_text_example: "' + self.environment.reset()[0] + '"')
        print('for see the training use this command: ' + 'tensorboard --logdir=' + self.login_directory)
        print('')

    def print_step_information(self, current_step_number, responses, rewards):
        rewards_on_cpu = np.array([reward.to('cpu') for reward in rewards])
        score_mean = np.mean(rewards_on_cpu)

        follow_instruction_mean = np.mean(np.array(self.follow_instruction_time))
        follow_instruction_sum = np.sum(np.array(self.follow_instruction_time))
        step_time_mean = np.mean(np.array(self.episode_time))
        instructions_generations = np.array(self.instructions_generations_time)
        learn = np.array(self.learn_time)
        batch = np.array(self.batch_time)

        print('--- Step ' + str(current_step_number) + ' information ---')
        print('score mean: ' + str(score_mean))

        print()
        print('batch: ' + str(batch) + 's')
        print('instructions generations : ' + str(instructions_generations) + 's')
        print('learn: ' + str(learn) + 's')
        print('follow instruction sum: ' + str(follow_instruction_sum) + 's')
        print('follow instruction mean: ' + str(follow_instruction_mean) + 's')
        print('episode time mean: ' + str(step_time_mean) + 's')
        print()

        print('best instruction score: ' + str(rewards_on_cpu[np.argmax(rewards_on_cpu)]))
        print('best instruction: ' + self.model_generates_instructions.decode(responses[np.argmax(rewards_on_cpu)]))

        random_index = random.randint(0, len(responses)-1)
        print('random instruction score: ' + str(rewards_on_cpu[random_index]))
        print('random instruction: ' + self.model_generates_instructions.decode(responses[random_index]))
        print('')

        self.writer.add_scalar(tag='reward_mean', scalar_value=score_mean, global_step=current_step_number)






