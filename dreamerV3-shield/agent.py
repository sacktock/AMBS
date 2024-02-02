import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

import behaviors
import jaxagent
import jaxutils
import nets
import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.shielding:
      self.safe_behavior = getattr(behaviors, config.safe_behavior)(
        self.wm, self.act_space, self.config, name='safe_behavior')
    else:
      self.safe_behavior = None
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    if self.config.shielding:
      return (
          self.wm.initial(batch_size),
          self.task_behavior.initial(batch_size),
          self.safe_behavior.initial(batch_size),
          self.expl_behavior.initial(batch_size))
    else:
      return (
          self.wm.initial(batch_size),
          self.task_behavior.initial(batch_size),
          self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    if self.config.shielding:
      (prev_latent, prev_action), task_state, expl_state, safe_state = state
    else:
      (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if self.config.shielding:
      safe_outs, safe_state = self.safe_behavior.policy(latent, safe_state)
    else:
      safe_outs, safe_state = None, None
    if mode == 'eval':
      outs = task_outs
      if self.config.shielding:
        prob = self.wm.shield(self.task_behavior.ac, latent, obs, self.config.imag_horizon)
        override = prob > self.config.shield_eps
        vec = jnp.expand_dims(override, axis=1)
        task_mask = 1.0 - vec 
        safe_mask = 0.0 + vec
        safe_outs['action'] = safe_outs['action'].sample(seed=nj.rng())
        task_outs['action'] = task_outs['action'].sample(seed=nj.rng())
        outs['action'] = task_outs['action'] * task_mask + safe_outs['action'] * safe_mask
      else:
        outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      if self.config.shielding:
        prob = self.wm.shield(self.expl_behavior.ac, latent, obs, self.config.imag_horizon)
        override = prob > self.config.shield_eps
        vec = jnp.expand_dims(override, axis=1)
        expl_mask = 1.0 - vec 
        safe_mask = 0.0 + vec
        safe_outs['log_entropy'] = safe_outs['action'].entropy()
        safe_outs['action'] = safe_outs['action'].sample(seed=nj.rng())
        expl_outs['log_entropy'] = expl_outs['action'].entropy()
        expl_outs['action'] = expl_outs['action'].sample(seed=nj.rng())
        outs['action'] = expl_outs['action'] * expl_mask + safe_outs['action'] * safe_mask
        outs['log_entropy'] = expl_outs['log_entropy'] * expl_mask + safe_outs['log_entropy'] * safe_mask
      else:
        outs['log_entropy'] = outs['action'].entropy()
        outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    if self.config.shielding:
      state = ((latent, outs['action']), task_state, expl_state, safe_state)
    else:
      state = ((latent, outs['action']), task_state, expl_state)

    if self.config.shielding and mode in ['eval', 'explore']:
      mets = {
        'prob': prob,
        'override': override,
      }
    else:
      mets = {}
    return outs, state, mets

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    if self.config.shielding:
      _, mets = self.safe_behavior.train(self.wm.imagine, start, context)
      metrics.update({'safe_' + key: value for key, value in mets.items()})
      if self.config.plpg:
        context.update({
          'safe_valuefn': self.safe_behavior.ac.critics['cost'],
        })
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update({'task_' + key: value for key, value in mets.items()})
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.safe_behavior is not None:
      mets = self.safe_behavior.report(data)
      report.update({f'safe_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    obs['safe_cont'] = 1.0 - (obs['cost'] > 0.0).astype(jnp.float32)
    obs['cost'] = jnp.clip(obs['cost'], 0.0, self.config.env.safetygym.cost_val)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    if config.shielding:
      self.heads.update({
        'cost': nets.MLP((), **config.cost_head, name='cost'),
        'safe_cont': nets.MLP((), **config.safe_cont_head, name='safe_cont')
      })
      discount = 1 - 1 / config.safety_horizon
      self._delta = config.env.safetygym.cost_val * pow(discount, self.config.shield_horizon)
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    first_safe_cont = (1.0 - (start['cost'] > 0.0)).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    safety_discount = 1 - 1 / self.config.safety_horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    if self.config.shielding:
      assert 'safe_cont' in self.heads.keys()
      safe_cont = self.heads['safe_cont'](traj).mode()
      traj['safe_cont'] = jnp.concatenate([first_safe_cont[None], safe_cont[1:]], 0)
      traj['safe_weight'] = jnp.cumprod(safety_discount * traj['safe_cont'], 0) / safety_discount
    return traj

  def shield(self, imag_behavior, latent, obs, horizon):
    dim = latent['deter'].shape[0]
    assert dim == obs['cost'].shape[0] == obs['cont'].shape[0]
    tile = lambda x : jnp.tile(x, [self.config.shield_samples] + [1]*len(x.shape)) 
    concat = lambda x : jnp.concatenate([tile(x[i]) for i in range(0, dim)], axis=0)
    first_cont = concat(obs['cont'])
    # Roll out the policy
    actor = imag_behavior.actor
    policy = lambda s: actor(sg(s)).sample(seed=nj.rng())
    keys = list(self.rssm.initial(1).keys())
    latent = {k: v for k, v in latent.items() if k in keys}
    start = {k : concat(v) for k, v in latent.items()}
    assert start['deter'].shape[0] == dim * self.config.shield_samples 
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    # Think about clipping the cost and/or value function
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    traj['cost'] = jnp.clip(self.heads['cost'](traj).mean()[1:], 0.0, self.config.env.safetygym.cost_val)
    discount = 1 - 1 / self.config.safety_horizon
    # Compute the discounted return
    if self.config.shield_bootstrap:
      valuefn = lambda tr: jnp.clip(imag_behavior.critics['cost'].valuefn(tr), 0.0, self.config.env.safetygym.cost_val)
    else:
      valuefn = None
    ret = disc_cost(traj, discount, valuefn)[0]
    x = (ret > self._delta).astype(jnp.float32)
    # Return the vector of probabilities
    prob = jnp.array([jnp.mean(x[i:i+self.config.shield_samples]) \
      for i in range(0, dim*self.config.shield_samples, self.config.shield_samples)])
    assert prob.shape[0] == dim
    return prob

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    metrics['cost_max_data'] = jnp.abs(data['cost']).max()
    metrics['cost_max_pred'] = jnp.abs(dists['cost'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cost' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cost'], data['cost'], 0.1)
      metrics.update({f'cost_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    if 'safe_cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['safe_cont'], data['safe_cont'], 0.5)
      metrics.update({f'safe_cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config, args={}):
    critics = {k: v for k, v in critics.items() if k in scales.keys()}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if k in scales.keys()}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    self.args = args if args else config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **self.args.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    if 'safe_valuefn' in context.keys():
      assert self.config.plpg
    safe_valuefn = context.get('safe_valuefn', None)
    def loss(start, safe_valuefn):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj, safe_valuefn=safe_valuefn)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, safe_valuefn, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj, safe_valuefn=None):
    metrics = {}
    advs = []
    if self.config.normalise_ret:
      total = sum(self.scales[k] for k in self.critics)
    else:
      total = 1.0

    for key, critic in self.critics.items():
      scale = self.scales[key]
      if scale == 0.0:
        continue
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      if key == 'extr' and self.config.plpg:
        assert safe_valuefn is not None
        td_error = safe_valuefn.compute_tds(traj, self.actor)
        td_error = jnp.clip(td_error, None, 0.0)
        exp_td_error = jnp.exp(td_error)
        clipped_exp_td_error = jnp.clip(exp_td_error, 1e-6, 1.0)
        metrics.update({
          'plpg_td_error_avg': jnp.mean(clipped_exp_td_error),
          'plpg_td_error_max': jnp.max(clipped_exp_td_error),
          'plpg_td_error_min': jnp.min(clipped_exp_td_error),
        })
        adv = ((normed_ret - normed_base) * clipped_exp_td_error) * scale / total
      else:
        adv = (normed_ret - normed_base) * scale / total
      if key == 'extr' and self.config.copt:
        assert 'cost' in self.critics.keys()
        costvalue_fn = lambda tr: jnp.clip(self.critics['cost'].valuefn(tr), 0.0, self.config.env.safetygym.cost_val)
        discount = 1 - 1 / self.config.safety_horizon
        traj['cost'] = self.critics['cost'].rewfn(traj)
        cost = disc_cost(traj, discount, costvalue_fn)
        weights = jnp_sigmoid(cost, scale=self.config.sigmoid_scale, 
          loc=self.config.env.safetygym.cost_val * pow(discount, self.config.shield_horizon))
        metrics.update({
          'copt_weights_avg': jnp.mean(weights),
          'copt_weights_max': jnp.max(weights),
          'copt_weights_min': jnp.min(weights),
        })
        adv = adv * (1.0 - sg(weights))
      advs.append(adv)
  
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config, args={}):
    self.rewfn = rewfn
    self.config = config
    self.args = args if args else config
    self.net = nets.MLP((), name='net', dims='deter', **self.args.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.args.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.args.slow_critic_fraction,
        self.args.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.args.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.args.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.args.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.args.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    if self.args.critic_cont_fn == 'cont':
      loss = (loss * sg(traj['weight'])).mean()
    elif self.args.critic_cont_fn == 'safe_cont':
      loss = (loss * sg(traj['safe_weight'])).mean()
    else:
      raise NotImplementedError(self.args.critic_cont_fn)
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.args.horizon
    if self.args.critic_cont_fn == 'cont':
      disc = traj['cont'][1:] * discount
    elif self.args.critic_cont_fn == 'safe_cont':
      disc = traj['safe_cont'][1:] * discount
    else:
      raise NotImplementedError(self.args.critic_cont_fn)
    value = self.valuefn(traj)
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]

  def compute_tds(self, traj, actor=None):
    # Compute undiscounted td errors
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    value = self.valuefn(traj)
    td_errors = []
    for t in range(len(rew)):
      td_errors.append(rew[t] + value[t+1] - value[t])
    td_errors = jnp.stack(td_errors)
    return td_errors

  def valuefn(self, traj):
    return self.net(traj).mean()


class TD3VFunction(nj.Module):

  def __init__(self, rewfn, config, args={}):
    self.rewfn = rewfn
    self.config = config
    self.args = args if args else config
    self.net_1 = nets.MLP((), name='net_1', dims='deter', **self.args.critic)
    self.slow_1 = nets.MLP((), name='slow_1', dims='deter', **self.args.critic)
    self.net_2 = nets.MLP((), name='net_2', dims='deter', **self.args.critic)
    self.slow_2 = nets.MLP((), name='slow_2', dims='deter', **self.args.critic)
    self.updater_1 = jaxutils.SlowUpdater(
        self.net_1, self.slow_1,
        self.args.slow_critic_fraction,
        self.args.slow_critic_update)
    self.updater_2 = jaxutils.SlowUpdater(
        self.net_2, self.slow_2,
        self.args.slow_critic_fraction,
        self.args.slow_critic_update)
    self.opt_1 = jaxutils.Optimizer(name='critic_1_opt', **self.args.critic_opt)
    self.opt_2 = jaxutils.Optimizer(name='critic_2_opt', **self.args.critic_opt)

  def train(self, traj, actor):
    def prepend(key, prefix=""):
      if key.startswith(prefix):
        return key
      else:
        return prefix + key
    target = sg(self.score(traj)[1])
    mets_1, metrics_1 = self.opt_1(self.net_1, self.loss, traj, target, self.net_1, self.slow_1, has_aux=True)
    metrics_1 = {prepend(k, prefix='1_'): v for k, v in metrics_1.items()}
    metrics_1.update(mets_1)
    self.updater_1()
    mets_2, metrics_2 = self.opt_2(self.net_2, self.loss, traj, target, self.net_2, self.slow_2, has_aux=True)
    metrics_2 = {prepend(k, prefix='2_'): v for k, v in metrics_2.items()}
    metrics_2.update(mets_2)
    self.updater_2()
    metrics = {**metrics_1, **metrics_2}
    return metrics

  def loss(self, traj, target, net, slow):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = net(traj)
    loss = -dist.log_prob(sg(target))
    if self.args.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(slow(traj).mean()))
    elif self.args.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.args.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    if self.args.critic_cont_fn == 'cont':
      loss = (loss * sg(traj['weight'])).mean()
    elif self.args.critic_cont_fn == 'safe_cont':
      loss = (loss * sg(traj['safe_weight'])).mean()
    else:
      raise NotImplementedError(self.args.critic_cont_fn)
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.args.horizon
    if self.args.critic_cont_fn == 'cont':
      disc = traj['cont'][1:] * discount
    elif self.args.critic_cont_fn == 'safe_cont':
      disc = traj['safe_cont'][1:] * discount
    else:
      raise NotImplementedError(self.args.critic_cont_fn)
    value = self.valuefn(traj)
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]

  def compute_tds(self, traj, actor=None):
    # Compute undiscounted td errors
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    value = self.valuefn(traj)
    td_errors = []
    for t in range(len(rew)):
      td_error = rew[t] + value[t+1] - value[t]
      td_errors.append(td_error)
    td_errors = jnp.stack(td_errors)
    return td_errors

  def valuefn(self, traj):
    value_1 = self.net_1(traj).mean()
    value_2 = self.net_2(traj).mean()
    value = jnp.minimum(value_1, value_2)
    return value


def disc_cost(traj, discount, valuefn=None):
  cost = traj['cost']
  assert len(cost) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
  
  disc = traj['cont'][1:] * discount
  if valuefn:
    value = valuefn(traj)
  else:
    value = jnp.zeros(cost.shape)
  vals = [value[-1]]
  interm = cost
  for t in reversed(range(len(disc))):
    vals.append(interm[t] + disc[t] * vals[-1])
  ret = jnp.stack(list(reversed(vals))[:-1])
  return ret

def jnp_sigmoid(arr, scale=1.0, loc=0.0):
  return jnp.where(arr >= 0.0,
    1.0 / (1.0 + jnp.exp(scale * (-1.0) * (arr - loc))),
    jnp.exp(scale * (arr - loc)) / (1.0 + jnp.exp(scale * (arr - loc))))


