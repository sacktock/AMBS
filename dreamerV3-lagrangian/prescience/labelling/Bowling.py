import gym
import numpy as np
from prescience.labelling import Labeller


class No_Hit(Labeller):
    def __init__(self, env):
        self.need_initial = True
        super().__init__(env)

    def reset(self):
        self.need_initial = True

    def label(self, obs, reward, done, info):
        if self.need_initial:
            # store the initial layout of the pins
            self.pins_initial_layout = np.sum(obs, axis=2)[119:159, 121:135]
            self.need_initial = False
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check if the ball has returned
        ball_returned = np.any(observation[170, 10:17] == 279)
        # check for no hit
        if ball_returned and \
                np.all(observation[119:159, 121:135] == self.pins_initial_layout):
            self.need_initial = True
            return True
        else:
            return False

    def save(self):
        return (self.need_initial, self.pins_initial_layout)

    def restore(self, state):
        self.need_initial, self.pins_initial_layout = state


class No_Strike(Labeller):
    def __init__(self, env):
        self.need_initial = False
        super().__init__(env)

    def reset(self):
        self.need_initial = False

    def label(self, obs, reward, done, info):
        # create a unique color map
        observation = np.sum(obs, axis=2)
        # check if the ball has returned
        ball_returned = np.any(observation[170, 10:17] == 279)
        # check for no strike
        if ball_returned and \
                np.any(observation[119:159, 121:135] == 279):
            return True
        else:
            return False
