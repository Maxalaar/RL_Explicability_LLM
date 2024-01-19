import random
from gymnasium import Env


class MinimalText(Env):
    def __init__(self):
        self.seed = 0
        random.seed(self.seed)
        self.description_environment = 'If the observation is 0, the response should be 1; if the observation is 0, the response should be 0.'
        self.actions_dictionary = {
            '0': 1,
            '1': 0,
        }
        self.actions_list = list(self.actions_dictionary.keys())
        self.max_number_step = 10
        self.current_value = None
        self.current_step = None

    def reset(self, **kwargs):
        self.current_step = 0
        self.current_value = random.randint(0, len(self.actions_list))
        return self.observation_to_text_observation(self.current_value), {}

    def step(self, action: str):
        if self.actions_dictionary[action] == self.current_value:
            if self.current_step < self.max_number_step:
                self.current_step += 1
                self.current_value = random.randint(0, len(self.actions_list))
                return self.observation_to_text_observation(self.current_value), 1/self.max_number_step, False, False, {}
            else:
                return None, 1/self.max_number_step, True, False, {}
        else:
            return None, 0, True, False, {}

    def close(self):
        pass

    def get_actions_tokens(self):
        return list(self.actions_dictionary.keys())

    def observation_to_text_observation(self, observation) -> str:
        return 'the observation is: ' + str(observation)
