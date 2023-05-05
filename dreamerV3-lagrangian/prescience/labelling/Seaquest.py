import gym
import numpy as np
from prescience.labelling import Labeller
from collections import deque


class Early_Surface(Labeller):
    def __init__(self, env):
        self.first_frame = True
        super().__init__(env)

    def reset(self):
        self.first_frame = True

    def label(self, obs, reward, done, info):
        # The first frame has to be discarded
        if self.first_frame:
            self.first_frame = False
            return False
        ram = self.env.ale.getRAM()
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check for early surface
        if np.any(observation[46, :] == 708) and \
                observation[186, 58] != 217 and \
                ram[102] > 4:
            return True
        else:
            return False

    def save(self):
        return self.first_frame

    def restore(self, state):
        self.first_frame = state


class Out_Of_Oxygen(Labeller):
    def __init__(self, env):
        self.first_frame = True
        self.bar_history = deque()
        super().__init__(env)

    def reset(self):
        self.first_frame = True
        self.bar_history = deque()

    def label(self, obs, reward, done, info):
        # the first frame has to be discarded
        if self.first_frame:
            self.first_frame = False
            return False
        # looking at the history of the oxygen bar
        self.bar_history.append(np.any(np.sum(obs, axis=2)[170:175, 49] == 0))

        if len(self.bar_history) > 20:
            self.bar_history.popleft()
            # create a unique color map of current frame
            observation = np.sum(obs, axis=2)
            if observation[170, 49] == 241 and sum(self.bar_history) > 0:
                return True
            else:
                return False
        else:
            return False

    def save(self):
        return (self.first_frame, self.bar_history)

    def restore(self, state):
        self.first_frame, self.bar_history = state