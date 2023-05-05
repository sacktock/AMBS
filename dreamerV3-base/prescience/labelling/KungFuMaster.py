import gym
from prescience.labelling import Labeller
import numpy as np
from collections import deque


class Energy_Loss(Labeller):
    def __init__(self, env):
        self.energy_bar = deque()
        super().__init__(env)

    def reset(self):
        self.energy_bar = deque()

    def label(self, obs, reward, done, info):
        # create a unique color map
        observation = np.sum(obs, axis=2)
        self.energy_bar.append(np.sum(observation[49:88, 41:46]))
        if len(self.energy_bar) > 2:
            self.energy_bar.popleft()
            # check for energy
            if self.energy_bar[1] < self.energy_bar[0]:
                return True
        return False

    def save(self):
        return self.energy_bar

    def restore(self, state):
        self.energy_bar = state
