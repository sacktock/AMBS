import gym
import time
from collections import deque


class Labeller():
    """Each labeller object corresponds to only one property"""

    def save(self):
        return None

    def restore(self, state):
        pass

    def reset(self):
        raise NotImplementedError

    def __init__(self, env):
        self.env = env

    def label(self, obs, reward, done, info):
        """Returns True if the safety property is violated, False otherwise"""
        raise NotImplementedError

    def test_random(self, pause=1, grace=30, max_steps=10000, speed=0.001):
        self.env.reset()
        done = False
        step = 0
        last_violation = 0
        while (not done) and step < max_steps:
            action = self.env.action_space.sample()
            _, reward, done, info = self.env.step(action)
            obs = self.env.render('rgb_array')
            self.env.render()
            step += 1
            if self.label(obs, reward, done, info):
                print('Property violated at step %d' % step)
                if step - last_violation > grace:
                    time.sleep(pause)
                    last_violation = step
            else:
                time.sleep(speed)
        self.env.close()

    def test_human(self, fps=60):
        from gym.utils.play import play  # Import here since it requires pygame and it is incompatible with python 3.8
        """This doesn't work if the environment has been wrapped with image preprocessing"""

        def callback_label(obs_t, obs_tp1, action, reward, done, info):
            if self.label(obs_tp1, reward, done, info):
                print('Property violated!')
            if done:
                self.reset()

        play(self.env, callback=callback_label, zoom=4, fps=fps)
