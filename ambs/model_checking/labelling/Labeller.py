import gym
import time
from collections import deque

class Labeller():
    """Labeller object corresponding to one atom"""

    def save(self):
        return None

    def restore(self, state):
        pass

    def reset(self):
        raise NotImplementedError

    def __init__(self, env):
        self.env = env

    def label(self, obs, reward, done, info):
        """Returns True if the atom holds from the current state False otherwise"""
        raise NotImplementedError
