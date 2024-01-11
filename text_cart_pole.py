import random

import gymnasium as gym
from gymnasium import Env


class TextCartPole(Env):
    def __init__(self):
        self.environment = gym.make('CartPole-v1')
        self.seed = 0
        self.max_reward = 500
        # Provide a set of instructions to solve the following task:
        self.description_environment = 'You control a Cart with a Pole on top; you must prevent it from falling. You can move from right to left to prevent the Pole from falling. Actions are, 0: Push cart to the left, 1: Push cart to the right. '
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
        else:
            action = 0

        observation, reward, terminated, truncated, information = self.environment.step(action)

        return self.observation_to_text_observation(observation), reward, terminated, truncated, information

    def close(self):
        self.environment.close()

    def get_actions_tokens(self):
        return self.actions_dictionary.keys()

    def observation_to_text_observation(self, observation) -> str:
        return self.description_observation + ' The current observation is : ' + str(observation)



