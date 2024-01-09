import time
from datetime import datetime
import torch
import numpy as np
import gymnasium
from trl import PPOTrainer, PPOConfig
import tensorflow as tf

from model_follows_instructions import ModelFollowsInstructions
from model_generates_instructions import ModelGeneratesInstructions


class Trainer:
    def __init__(self, model_generates_instructions: ModelGeneratesInstructions, model_follows_instructions: ModelFollowsInstructions, environment, batch_size: int = 1, ):
        self.model_generates_instructions: ModelGeneratesInstructions = model_generates_instructions
        self.model_follows_instructions: ModelFollowsInstructions = model_follows_instructions

        self.environment: gymnasium.Env = environment

        self.batch_size: int = batch_size
        self.configuration = PPOConfig(batch_size=self.batch_size)
        self.trainer = PPOTrainer(self.configuration, self.model_generates_instructions.model, self.model_generates_instructions.model_reference, self.model_generates_instructions.tokenizer)

        self.login_directory = './login/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = tf.summary.create_file_writer(self.login_directory)

        self.print_initialisation_information()

    def step(self):
        instruction: str = self.model_generates_instructions.instruction_generation(self.environment.description_environment)
        return self.model_generates_instructions.encode(self.environment.description_environment), self.model_generates_instructions.encode(instruction), torch.tensor(self.episode(instruction))

    def episode(self, instruction: str):
        total_reward: float = 0.0
        observation, info = self.environment.reset()
        terminated = False

        while not terminated:
            action = self.model_follows_instructions.follow_instructions(instruction, observation)
            observation, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward

        self.environment.close()

        return total_reward/self.environment.max_reward

    def learn(self, number_steps: int):
        for i in range(number_steps):
            queries = []
            responses = []
            rewards = []
            steps_time = []

            for j in range(self.batch_size):
                start_time = time.time()
                queri, response, reward = self.step()
                end_time = time.time()
                steps_time.append(end_time - start_time)

                queries.append(queri[0])
                responses.append(response[0])
                rewards.append(reward)

            self.trainer.step(queries, responses, rewards)
            self.print_step_information(i, responses, rewards, steps_time)

    def print_initialisation_information(self):
        print('The model for generates instructions:')
        print('id: ' + self.model_generates_instructions.model_id)
        print('device: ' + self.model_generates_instructions.device)
        print('memory: ' + str(self.model_generates_instructions.get_memory_footprint()))
        print('')
        print('The model for generates instructions:')
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

    def print_step_information(self, current_step_number, responses, rewards, steps_time):
        score_mean = np.mean(np.array(rewards))
        step_time_mean = np.mean(np.array(steps_time))

        print('Step ' + str(current_step_number) + ' information :')
        print('score mean: ' + str(score_mean))
        print('step time mean: ' + str(step_time_mean) + 's')
        print('best instruction: ' + self.model_generates_instructions.decode(responses[np.argmax(np.array(rewards))]))
        print('')
        with self.writer.as_default():
            tf.summary.scalar('reward_mean', score_mean, step=current_step_number)





