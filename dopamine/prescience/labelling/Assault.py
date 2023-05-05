import gym
from prescience.labelling import Labeller
import numpy as np
from collections import deque


class Overheat(Labeller):
    def __init__(self, env):
        self.lives_history = deque()
        self.over_heat_happened = False
        super().__init__(env)

    def reset(self):
        self.lives_history = deque()
        self.over_heat_happened = False

    def label(self, obs, reward, done, info):
        self.lives_history.append(info['ale.lives'])
        if len(self.lives_history) > 2:
            self.lives_history.popleft()
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for overheat
        if np.all(observation[229:237, 156:160] != 0):
            self.over_heat_happened = True
        if self.over_heat_happened and self.lives_history[0] > info['ale.lives']:
            self.over_heat_happened = False
            return True
        else:
            return False

    def save(self):
        return (self.lives_history, self.over_heat_happened)

    def restore(self, state):
        self.lives_history, self.over_heat_happened = state
