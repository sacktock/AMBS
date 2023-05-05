import gym
from prescience.labelling import Labeller
import numpy as np
from collections import deque


class Dynamite(Labeller):
    def __init__(self, env):
        self.lives_history = deque()
        self.dynamite_exploded = False
        self.counter = 0
        super().__init__(env)

    def reset(self):
        self.lives_history = deque()
        self.dynamite_exploded = False
        self.counter = 0

    def label(self, obs, reward, done, info):
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for dynamite explosion
        if np.all(observation[15, 8:160] == 576):
            self.dynamite_exploded = True
            print('Boom!')
            self.counter = 0
        # check for 31 frames
        if self.dynamite_exploded and self.counter < 36:
            self.counter += 1
        elif self.counter >= 36:
            self.counter = 0
            self.dynamite_exploded = False
        self.lives_history.append(info['ale.lives'])
        if len(self.lives_history) > 35:
            self.lives_history.popleft()
        # check for death after dynamite explosion
        if self.dynamite_exploded and self.lives_history[0] > info['ale.lives']:
            self.dynamite_exploded = False
            return True
        else:
            return False

    def save(self):
        return (self.lives_history, self.dynamite_exploded, self.counter)

    def restore(self, state):
        self.lives_history, self.dynamite_exploded, self.counter = state


class Out_Of_Power(Labeller):
    def __init__(self, env):
        self.lives_history = deque()
        super().__init__(env)

    def reset(self):
        self.lives_history = deque()

    def label(self, obs, reward, done, info):
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for dynamite explosion
        if np.all(observation[15, 8:160] == 576):
            self.dynamite_exploded = True
            self.counter = 0
        # check for 31 frames
        if self.dynamite_exploded and self.counter < 36:
            self.counter += 1
        elif self.counter >= 36:
            self.counter = 0
            self.dynamite_exploded = False
        self.lives_history.append(info['ale.lives'])
        if len(self.lives_history) > 35:
            self.lives_history.popleft()
        # check for death after dynamite explosion
        if self.dynamite_exploded and self.lives_history[0] > info['ale.lives']:
            self.dynamite_exploded = False
            return True
        else:
            return False
