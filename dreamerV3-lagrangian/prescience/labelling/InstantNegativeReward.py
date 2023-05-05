import gym
import numpy as np
from prescience.labelling import Labeller


class Instant_Negative_Reward(Labeller):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        pass

    def label(self, obs, reward, done, info):
        if reward < 0:
            return True
        else:
            return False
