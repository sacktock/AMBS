import gym
from prescience.labelling import Labeller
import numpy as np
from collections import deque


class Fuel(Labeller):
    def __init__(self, env):
        self.lives_history = deque()
        self.fuel_cir = False
        self.fuel_area = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 0,
                                    0, 0, 0, 0, 513, 513, 513, 513, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 513, 513,
                                    513, 0, 0, 0, 513, 513, 513, 513, 0]])
        super().__init__(env)

    def reset(self):
        self.lives_history = deque()
        self.fuel_cir = False
        self.fuel_area = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 0,
                                    0, 0, 0, 0, 513, 513, 513, 513, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 0,
                                    0, 0, 0, 513, 513, 0, 0, 513, 513],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 513, 513, 513, 513, 513,
                                    513, 0, 0, 0, 513, 513, 513, 513, 0]])

    def label(self, obs, reward, done, info):
        self.lives_history.append(info['ale.lives'])
        if len(self.lives_history) > 3:
            self.lives_history.popleft()
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for fuel
        if np.all(observation[17:24, 72:94] == self.fuel_area):
            self.fuel_cir = True
        if self.fuel_cir and self.lives_history[0] > info['ale.lives']:
            self.fuel_cir = False
            return True
        else:
            return False

    def save(self):
        return (self.fuel_cir, self.lives_history)

    def restore(self, state):
        self.fuel_cur, self.lives_history = state
