import gym
import numpy as np
from prescience.labelling import Labeller


class Below_Reward(Labeller):
    def __init__(self, env, threshold, count_pos=True, count_neg=True):
        self.total_reward = 0
        self.threshold = threshold
        self.count_pos = count_pos
        self.count_neg = count_neg
        super().__init__(env)

    def reset(self):
        self.total_reward = 0

    def label(self, obs, reward, done, info):
        if (reward > 0 and self.count_pos) or (reward < 0 and self.count_neg):
            self.total_reward += reward
        if done and self.total_reward < self.threshold:
            return True
        else:
            return False

    def save(self):
        return self.total_reward

    def restore(self, state):
        self.total_reward = state
