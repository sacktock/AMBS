import gym
import numpy as np
from ambs.model_checking.labelling import Labeller
from collections import deque

class Surface(Labeller):
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
        ram = self.env.unwrapped.ale.getRAM()
        
        # create a unique color map
        #observation = np.sum(obs, axis=2)
        #print('Depth:', ram[97], 'Oxygen:', ram[102], 'Divers:', ram[62], 'Flashing surface:', np.any(observation[46, :] == 708), 'Flashing any:', np.any(observation == 708))
        #print(((not np.any(observation[46, :] == 708)) or (ram[97] == 13)))
        if ram[97] == 13:
            return True
        else:
            return False

    def save(self):
        return self.first_frame

    def restore(self, state):
        self.first_frame = state

class Diver(Labeller):
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
        ram = self.env.unwrapped.ale.getRAM()

        # create a unique color map
        observation = np.sum(obs, axis=2)

        if ram[62] > 0:
            return True
        else:
            return False

    def save(self):
        return self.first_frame

    def restore(self, state):
        self.first_frame = state

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
        ram = self.env.unwrapped.ale.getRAM()
        
        # create a unique color map
        observation = np.sum(obs, axis=2)
        #print('Depth:', ram[97], 'Oxygen:', ram[102], 'Divers:', ram[62], 'Flashing surface:', np.any(observation[46, :] == 708), 'Flashing any:', np.any(observation == 708))
        #print(((not np.any(observation[46, :] == 708)) or (ram[97] == 13)))
        if np.any(observation[46, :] == 708):
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


class Death(Labeller):
    def __init__(self, env):
        self.env = env
        self.current_lives = self.env.unwrapped.ale.lives()

    def reset(self):
        self.current_lives = self.env.unwrapped.ale.lives()

    def label(self, obs, reward, done, info):
        new_lives = info['ale.lives']
        if new_lives < self.current_lives:
            self.current_lives = new_lives
            return True
        else:
            self.current_lives = new_lives
            return False

    def save(self):
        return self.current_lives

    def restore(self, state):
        self.current_lives = state