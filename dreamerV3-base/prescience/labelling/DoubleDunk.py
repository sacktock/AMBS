import gym
import numpy as np
from prescience.labelling import Labeller


class Out_Of_Bounds(Labeller):
    def __init__(self, env):
        self.need_initial = True
        super().__init__(env)

    def reset(self):
        self.need_initial = True

    def label(self, obs, reward, done, info):
        observation = np.sum(obs, axis=2)
        if np.all(observation[209:216, 95] == 309):
            return True
        else:
            return False


class Shoot_Bf_Clear(Labeller):
    def __init__(self, env):
        self.need_initial = True
        super().__init__(env)

    def reset(self):
        self.need_initial = True

    def label(self, obs, reward, done, info):
        if self.need_initial:
            self.look_out_for_sbc = 0
            self.need_initial = False
        observation = np.sum(obs, axis=2)
        if np.all(observation[209:214, 103] == 309):
            self.look_out_for_sbc = 1
            return False
        elif np.all(observation[209:216, 58:60] == 309) and self.look_out_for_sbc == 1:
            self.look_out_for_sbc = 0
            return True
        else:
            self.look_out_for_sbc = 0
            return False

    def save(self):
        return (self.need_initial, self.look_out_for_sbc)

    def restore(self, state):
        self.need_initial, self.look_out_for_sbc = state
