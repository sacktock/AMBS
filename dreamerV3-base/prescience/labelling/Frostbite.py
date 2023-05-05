import gym
import numpy as np
from prescience.labelling import Labeller


class Freezing(Labeller):
    def __init__(self, env):
        # store the freezing timer pixels
        self.timer_pixels_when_zero = np.array([279, 279, 279, 279, 279, 279, 279, 279, 279, 528, 528, 528, 528, 279])
        for _ in range(4):
            self.timer_pixels_when_zero = np.vstack(
                (self.timer_pixels_when_zero, [279, 279, 279, 279, 279, 279, 279, 279, 528, 528, 279, 279, 528, 528]))
        super().__init__(env)

    def reset(self):
        pass

    def label(self, obs, reward, done, info):
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for freezing
        if np.all(observation[22:27, 23:37] == self.timer_pixels_when_zero):
            return True
        else:
            return False
