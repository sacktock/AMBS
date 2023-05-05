import gym
import numpy as np
from prescience.labelling import Labeller


class Hit(Labeller):
    def __init__(self, env):
        self.first_frame = True
        super().__init__(env)

    def reset(self):
        self.first_frame = True

    def label(self, obs, reward, done, info):
        # The first frame doesn't contain the chicken
        if self.first_frame:
            self.first_frame = False
            return False
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # find the chicken location
        try:
            agent_row = np.max(np.where((observation[:, 48] == 588)))
        except:
            return False
        # check for the hit or win
        if observation[agent_row - 5, 45] == 588 and \
                agent_row - 5 != 104 and \
                agent_row - 5 != 102:
            return True
        else:
            return False

    def save(self):
        return self.first_frame

    def restore(self, state):
        self.first_frame = state
