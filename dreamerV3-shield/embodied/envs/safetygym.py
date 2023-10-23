import embodied
import numpy as np
import gym
import safety_gym
from PIL import Image

class SafetyGym(embodied.Env):

    LOCK = None

    def __init__(self, name, repeat=2, size=(64, 64), length=1000, vision=False, render_type='rgb_image', cost_val=1.0):
        assert size[0] == size[1]
        assert render_type in ('rgb_image', 'binary_image')
        if self.LOCK is None:
            import multiprocessing as mp
            mp = mp.get_context('spawn')
            self.LOCK = mp.Lock()
        with self.LOCK:
            self._env = gym.make(name)
        if length:
            self._env._max_episode_steps = length
        if vision:
            self._env.unwrapped.vision_size = size
            self._env.unwrapped.observe_vision = True
            self._env.unwrapped.vision_render = False
            obs_vision_swap = self._env.unwrapped.obs_vision

            from PIL import ImageOps

            def render_obs(fake=True):
                if fake:
                    return np.ones(())
                else:
                    image = Image.fromarray(np.array(obs_vision_swap() * 255, dtype=np.uint8,
                                                    copy=False))
                    image = np.asarray(ImageOps.flip(image))
                    return image

            self._env.unwrapped.obs_vision = render_obs

            def safety_gym_render(mode, **kwargs):
                if mode in ['human', 'rgb_array']:
                    # Use regular rendering
                    return self._env.unwrapped.render(mode, camera_id=3, **kwargs)
                elif mode == 'vision':
                    return render_obs(fake=False)
                else:
                    raise NotImplementedError

            self._env.render = safety_gym_render

        self._done = True
        self._length = length
        self._vision = vision
        self._repeat = repeat
        self._size = size
        self._cost_val = cost_val
        self._type = render_type
        self._render_kwargs = {'mode': 'vision'}

    @property
    def obs_space(self):
        if not self._vision:
            shape = self._env.observation_space.shape
            return {
                'image': embodied.Space(np.float32, shape),
                'reward': embodied.Space(np.float32),
                'cost': embodied.Space(np.float32),
                'is_first': embodied.Space(bool),
                'is_last': embodied.Space(bool),
                'is_terminal': embodied.Space(bool),
            }
        else:
            shape = self._size + (1 if self._type == 'binary_image' else 3,)
            return {
                'image': embodied.Space(np.uint8, shape),
                'reward': embodied.Space(np.float32),
                'cost': embodied.Space(np.float32),
                'is_first': embodied.Space(bool),
                'is_last': embodied.Space(bool),
                'is_terminal': embodied.Space(bool),
            }

    @property
    def act_space(self):
        shape = self._env.action_space.shape
        return {
            'action': embodied.Space(np.float32, shape, -1.0, 1.0),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            with self.LOCK:
                obs = self._reset()
            self._done = False
            return self._obs(None if self._vision else obs, 0.0, 0.0, is_first=True)

        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        total = 0.0
        total_cost = 0.0
        obs = None
        for repeat in range(self._repeat):
            obs, reward, done, info = self._env.step(action['action'])
            total += reward
            total_cost += bool(info.get('cost', 0)) * self._cost_val 
            self._done = done
            if done:
                break
        
        assert obs is not None
        return self._obs(
            None if self._vision else obs,
            total,
            total_cost,
            is_last=self._done,
            is_terminal=self._done,
        )

    def _reset(self):
        return self._env.reset()

    def _obs(self, obs, reward, cost, is_first=False, is_last=False, is_terminal=False):
        if obs is None:
            assert self._vision
            image = self._env.render(**self._render_kwargs)
            image = Image.fromarray(image)

            if image.size != self._size:
                image = image.resize(self._size, Image.BILINEAR)
            if self._type == 'binary_image':
                image = image.convert('L')
            obs = np.array(image, copy=False)
            obs = np.clip(obs, 0, 255).astype(np.uint8) 

        return dict(
            image=obs,
            reward=reward,
            cost=cost,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def render(self):
        return self._env.render()
        
    def close(self):
        return self._env.close()
