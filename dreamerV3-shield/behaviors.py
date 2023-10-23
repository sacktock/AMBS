import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import embodied
import agent
import expl
import ninjax as nj
import jaxutils


class Greedy(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    critics = {}
    scales = {'extr': 1.0}
    if config.shielding and config.shield_bootstrap:
      scales.update({'cost': 0.0})
    if config.penalty_coeff:
      scales.update({'penl': config.penalty_coeff})
    for key, scale in scales.items():
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        if config.critic_type == 'vfunction':
          critics[key] = agent.VFunction(rewfn, config, name=key)
        else:
          raise NotImplementedError(config.critic_type)
      if key == 'penl':
        penlfn = lambda s: jnp.clip(wm.heads['cost'](s).mean()[1:], 0.0, self.config.env.safetygym.cost_val) * (-1.0)
        if config.use_safety_critic_params:
          args = embodied.Config(
          critic=self.config.safety_critic, critic_opt=self.config.safety_critic_opt,
          slow_critic_fraction=self.config.slow_safety_critic_fraction,
          slow_critic_update=self.config.slow_safety_critic_update,
          critic_slowreg=self.config.safety_critic_slowreg,
          critic_cont_fn=self.config.safety_critic_cont_fn,
          horizon=self.config.safety_horizon
          )
        else:
          args = None
        if config.penl_critic_type == 'td3vfunction':
          critics[key] = agent.VFunction(penlfn, config, args, name=key)
        elif config.penl_critic_type == 'vfunction':
          critics[key] = agent.VFunction(penlfn, config, args, name=key)
        else:
          raise NotImplementedError(config.critic_type)
      if key == 'cost':
        costfn = lambda s: jnp.clip(wm.heads['cost'](s).mean()[1:], 0.0, self.config.env.safetygym.cost_val)
        args = embodied.Config(
          critic=self.config.safety_critic, critic_opt=self.config.safety_critic_opt,
          slow_critic_fraction=self.config.slow_safety_critic_fraction,
          slow_critic_update=self.config.slow_safety_critic_update,
          critic_slowreg=self.config.safety_critic_slowreg,
          critic_cont_fn=self.config.safety_critic_cont_fn,
          horizon=self.config.safety_horizon
          )
        if config.safety_critic_type == 'td3vfunction':
          critics[key] = agent.TD3VFunction(costfn, config, args, name=key)
        elif config.safety_critic_type == 'vfunction':
          critics[key] = agent.VFunction(costfn, config, args, name=key)
        else:
          raise NotImplementedError(config.safety_critic_type) 
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Safe(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    rewfn = lambda s: jnp.clip(wm.heads['cost'](s).mean()[1:], 0.0, self.config.env.safetygym.cost_val) * (-1.0)
    args = embodied.Config(
      critic=self.config.critic, critic_opt=self.config.safe_critic_opt,
      slow_critic_fraction=self.config.slow_critic_fraction,
      slow_critic_update=self.config.slow_critic_update,
      critic_slowreg=self.config.critic_slowreg,
      critic_cont_fn=self.config.critic_cont_fn,
      horizon=self.config.horizon
      )
    if config.critic_type == 'vfunction':
      critics = {'cost': agent.VFunction(rewfn, config, args, name='critic')}
    else:
      raise NotImplementedError(config.critic_type)
    args = embodied.Config(
      actor_opt=self.config.safe_actor_opt
    )
    self.ac = agent.ImagActorCritic(
        critics, {'cost': 1.0}, act_space, config, args, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state, {}

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    if config.shielding and config.shield_bootstrap:
      raise NotImplementedError('config.safety_critic for explore behavior')
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}
