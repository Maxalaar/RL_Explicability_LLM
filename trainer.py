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

from accelerate import Accelerator
import os


class Trainer:
    def __init__(self, model_generates_instructions: ModelGeneratesInstructions, model_follows_instructions: ModelFollowsInstructions, environment, batch_size: int = 1):
        self.model_generates_instructions: ModelGeneratesInstructions = model_generates_instructions
        self.model_follows_instructions: ModelFollowsInstructions = model_follows_instructions

        self.environment: gymnasium.Env = environment

        self.batch_size: int = batch_size
        self.configuration = PPOConfig(batch_size=self.batch_size)
        self.trainer = PPOTrainer(self.configuration, self.model_generates_instructions.model, self.model_generates_instructions.model_reference, self.model_generates_instructions.tokenizer)

        self.login_directory = './login/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(logdir=self.login_directory)

        self.print_initialisation_information()

        self.instruction_generation_time = None
        self.follow_instructions_time = None
        self.step_time = None
        self.batch_time = None

    def step(self):
        start_instruction_generation_time_time = time.time()
        instruction: str = self.model_generates_instructions.instruction_generation(self.environment.description_environment)
        end_instruction_generation_time_time = time.time()
        self.instruction_generation_time.append(end_instruction_generation_time_time - start_instruction_generation_time_time)
        
        return self.model_generates_instructions.encode(self.environment.description_environment), self.model_generates_instructions.encode(instruction), torch.tensor(self.episode(instruction))

    def episode(self, instruction: str):
        total_reward: float = 0.0
        observation, info = self.environment.reset()
        terminated = False

        while not terminated:
            start_follows_instructions_time = time.time()
            action = self.model_follows_instructions.follow_instructions(instruction, observation)
            end_follows_instructions_time = time.time()
            self.follow_instructions_time.append(end_follows_instructions_time - start_follows_instructions_time)

            
            observation, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward

        self.environment.close()

        return total_reward

    def learn(self, number_steps: int):
        for i in range(number_steps):
            queries = []
            responses = []
            rewards = []
            
            self.instruction_generation_time = []
            self.follow_instructions_time = []
            self.step_time = []
            self.batch_time = []

            start_batch_time = time.time()
            for j in range(self.batch_size):
                
                start_step_time = time.time()
                queri, response, reward = self.step()
                end_step_time = time.time()
                self.step_time.append(end_step_time - start_step_time)

                queries.append(queri[0])
                responses.append(response[0])
                rewards.append(reward)
            end_batch_time = time.time()
            self.batch_time.append(end_batch_time - start_batch_time)

            self.trainer.step(queries, responses, rewards)
            self.print_step_information(i, responses, rewards)

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

        instruction_generation_mean = np.mean(np.array(self.instruction_generation_time))
        follow_instructions_mean = np.mean(np.array(self.follow_instructions_time))
        step_time_mean = np.mean(np.array(self.step_time))
        batch_time_mean = np.mean(np.array(self.batch_time))
        

        print('--- Step ' + str(current_step_number) + ' information ---')
        print('score mean: ' + str(score_mean))
        # print('rewards: ' + str(rewards_on_cpu))
        # print('index max: ' + str(np.argmax(rewards_on_cpu)))

        print()
        print('instruction_generation_mean: ' + str(instruction_generation_mean) + 's')
        print('follow_instructions_mean: ' + str(follow_instructions_mean) + 's')
        print('step_time_mean: ' + str(step_time_mean) + 's')
        print('batch_time_mean: ' + str(batch_time_mean) + 's')
        print()

        print('best instruction score: ' + str(rewards_on_cpu[np.argmax(rewards_on_cpu)]))
        print('best instruction: ' + self.model_generates_instructions.decode(responses[np.argmax(rewards_on_cpu)]))
        random_index = random.randint(0, len(responses)-1)
        print('random instruction score: ' + str(rewards_on_cpu[random_index]))
        print('random instruction: ' + self.model_generates_instructions.decode(responses[random_index]))
        print('')
        self.writer.add_scalar(tag='reward_mean', scalar_value=score_mean, global_step=current_step_number)






