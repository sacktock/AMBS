from ambs import embodied
import numpy as np
import gym
from ambs.model_checking.labelling import get_property
import importlib
from itertools import compress

from ambs.model_checking.dfa import Atom
from ambs.model_checking.dfa import Truth
from ambs.model_checking.dfa import And
from ambs.model_checking.dfa import Or
from ambs.model_checking.dfa import Neg
from ambs.model_checking.dfa import Implies
from ambs.model_checking.dfa import DFA
from ambs.model_checking.dfa import Cost_Function

class LabellingEnv(gym.Wrapper):
    """Adds a label from a labeller to the info returned at each step"""

    def __init__(self, env, labellers):
        super().__init__(env)
        self.env = env
        self.labellers = labellers

    def reset(self):
        obs = self.env.reset()
        for labeller in self.labellers:
            labeller.reset()
        return obs

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info['ale.lives'] = self.env.unwrapped.ale.lives()
        done = term or trunc
        labels = [labeller.label(obs, reward, done, info) for labeller in self.labellers]
        info['labels'] = labels
        return obs, reward, term, trunc, info

class Seaquest(embodied.Env):

  LOCK = None

  def __init__(
      self, repeat=4, size=(84, 84), gray=True, noops=0, lives='unused',
      sticky=True, actions='all', length=108000, resize='opencv', seed=None, property_file="", cost_val=1.0, reset_on_death=False):
    assert size[0] == size[1]
    assert lives in ('unused', 'discount', 'reset'), lives
    assert actions in ('all', 'needed'), actions
    assert resize in ('opencv', 'pillow'), resize
    if self.LOCK is None:
      import multiprocessing as mp
      mp = mp.get_context('spawn')
      self.LOCK = mp.Lock()
    self._resize = resize
    if self._resize == 'opencv':
      import cv2
      self._cv2 = cv2
    if self._resize == 'pillow':
      from PIL import Image
      self._image = Image
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    name = 'seaquest'
    env_id = ("ALE/"+"".join(word.capitalize() for word in name.split('_'))+"-v5")
    self._repeat = repeat
    self._size = size
    self._gray = gray
    self._noops = noops
    self._lives = lives
    self._sticky = sticky
    self._length = length
    self._random = np.random.RandomState(seed)
    self._cost_val = cost_val
    self._reset_on_death = reset_on_death
    with self.LOCK:
      self._env = gym.make(
          env_id,
          obs_type='rgb',
          frameskip=1, repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=(actions == 'all'))
    assert self._env.unwrapped.get_action_meanings()[0] == 'NOOP'
    # labelling fn
    self._atomic_propositions = ['surface', 'diver', 'early-surface', 'out-of-oxygen', 'death']
    labellers = [get_property(self._env, 'seaquest', label) for label in self._atomic_propositions]
    self._env = LabellingEnv(self._env, labellers)
    self._property = property_file
    ctor = f'ambs.model_checking.properties.{self._property}:cost_function'
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    self._cost_function = ctor
    ctor = f'ambs.model_checking.properties.{self._property}:automaton'
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    self._automaton = ctor
    shape = self._env.observation_space.shape
    self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
    self._ale = self._env.unwrapped.ale
    self._last_lives = None
    self._done = True
    self._step = 0

  @property
  def obs_space(self):
    shape = self._size + (1 if self._gray else 3,)
    return {
        'image': embodied.Space(np.uint8, shape),
        'reward': embodied.Space(np.float32),
        'cost': embodied.Space(np.float32),
        'violation': embodied.Space(bool),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      with self.LOCK:
        self._reset()
      self._done = False
      self._step = 0
      return self._obs(0.0, 0.0, False, is_first=True)
    total = 0.0
    total_cost = 0.0
    dead = False
    labels = set()
    for repeat in range(self._repeat):
      _, reward, over, _, info = self._env.step(action['action'])

      self._step += 1
      total += reward
      if repeat == self._repeat - 2:
        self._buffer[1] = self._screen()

      current = self._ale.lives()
      if current < self._last_lives:
        self._last_lives = current
        if self._reset_on_death:
          self._cost_function.reset()
          self._automaton.reset()
          labels = set(compress(self._atomic_propositions, info['labels']))
        else:
          labels.update(set(compress(self._atomic_propositions, info['labels'])))
        if self._lives != 'unused':
          dead = True
          break
      else:
        labels.update(set(compress(self._atomic_propositions, info['labels'])))
      if over:
        break
    cost, _ = self._cost_function.step(labels)
    violation, _ = self._automaton.step(labels)
    total_cost += cost * self._cost_val
    if not self._repeat:
      self._buffer[1][:] = self._buffer[0][:]
    self._buffer[0] = self._screen()
    self._done = over or (self._length and self._step >= self._length)
    return self._obs(
        total,
        total_cost,
        violation,
        is_last=self._done or (dead and self._lives == 'reset'),
        is_terminal=dead or over)

  def _reset(self):
    self._env.reset()
    self._cost_function.reset()
    self._automaton.reset()
    if self._noops:
      for _ in range(self._random.randint(self._noops)):
         _, _, dead, _, _ = self._env.step(0)
         if dead:
           self._env.reset()
    self._last_lives = self._ale.lives()
    self._buffer[0] = self._screen()
    self._buffer[1].fill(0)

  def _obs(self, reward, cost, violation, is_first=False, is_last=False, is_terminal=False):
    np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
    image = self._buffer[0]
    if image.shape[:2] != self._size:
      if self._resize == 'opencv':
        image = self._cv2.resize(
            image, self._size, interpolation=self._cv2.INTER_AREA)
      if self._resize == 'pillow':
        image = self._image.fromarray(image)
        image = image.resize(self._size, self._image.NEAREST)
        image = np.array(image)
    if self._gray:
      weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
      image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
      image = image[:, :, None]
    return dict(
        image=image,
        reward=reward,
        cost=cost,
        violation=violation,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )

  def _screen(self):
    return np.array(self._ale.getScreenRGB()).astype(np.uint8)

  def close(self):
    return self._env.close()
