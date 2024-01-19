import random

import gymnasium as gym
from gymnasium import Env


class TextCartPole(Env):
    def __init__(self):
        self.environment = gym.make('CartPole-v1')
        self.seed = 0
        self.max_reward = 500
        # Provide a set of instructions to solve the following task:
        # self.description_environment = 'You are in control of a cart equipped with a pole on its top, and your objective is to prevent the pole from falling. You have the ability to move the cart from right to left, strategically maintaining balance. Your actions are limited to two choices: 0 for pushing the cart to the left and 1 for pushing it to the right. At each decision point, you are provided with crucial information, including the current Cart Position (min -4.8, max +4.8), Cart Velocity (min -inf, max +inf), Pole Angle (min -0.4, max +0.4), and Pole Angular Velocity (min -inf, max +inf). Now, let\'s outline the pseudocode instructions to successfully address this task:'
        self.description_environment = 'Controlling a cart with a pole on top to prevent the pole from falling, based on the provided observation inputs (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity) and output actions (0 for pushing the cart to the left and 1 for pushing it to the right).'
        self.description_observation = 'The observation is a 4-dimensional vector that breaks down as follows: (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity). '
        self.actions_dictionary = {
            '0': 0,
            'left': 0,
            'Left': 0,
            '1': 1,
            'right': 1,
            'Right': 1,
        }

    def reset(self, **kwargs):
        if self.seed is None:
            observation, information = self.environment.reset()
        else:
            observation, information = self.environment.reset(seed=self.seed)
        return self.observation_to_text_observation(observation), information

    def step(self, action: str):
        if action is not None:
            action: int = self.actions_dictionary[action]
            observation, reward, terminated, truncated, information = self.environment.step(action)
            return self.observation_to_text_observation(observation), 1, terminated, truncated, information
        else:
            return None, 0, True, False, {}

    def close(self):
        self.environment.close()

    def get_actions_tokens(self):
        return list(self.actions_dictionary.keys())

    def observation_to_text_observation(self, observation) -> str:
        # return self.description_observation + ' The current observation is : ' + str(observation)
        return 'Cart Position:' + str(observation[0]) + ', Cart Velocity:' + str(observation[1]) + ', Pole Angle:' + str(observation[2]) + ', Pole Angular Velocity:' + str(observation[3])



