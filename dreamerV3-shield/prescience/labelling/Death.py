import gym
from prescience.labelling import Labeller


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
