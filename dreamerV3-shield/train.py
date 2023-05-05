import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

import embodied
from embodied import wrappers
import numpy as np
import re
import agent as agt

from prescience.labelling.properties import prop_map

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def make_envs(config, **overrides):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))

def make_env(config, **overrides):
    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    suite, task = config.task.split('_', 1)
    ctor = {
        'atari': 'embodied.envs.atari:Atari',
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)

def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
    env = wrappers.ExpandScalars(env)
    env = wrappers.TimeLimit(env, args.length, args.reset)
    return env

def main(argv=None):
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    suite, task = config.task.split('_', 1)
    assert suite in ['atari'], f'Suite {suite} is not supported for model checking.'
    assert task in prop_map.keys(), f'Environment {config.task} not supported for model checking.'
    safety_labels = config.env.atari.labels if hasattr(config.env.atari, 'labels') else []
    assert safety_labels is not [], "No safety labels specified!"

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    # frame skip multiplier
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ], multiplier)

    print(f'Training on {config.task} with safety labels {safety_labels}')

    is_eval = False
    rate_limit = False
    directory = logdir / 'replay'
    assert config.replay == 'uniform' or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'reverb':
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == 'chunks':
        replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)

    cleanup = []
    env = make_envs(config)
    cleanup.append(env)
    agent = agt.Agent(env.obs_space, env.act_space, step, config)

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print('Observation space:', embodied.format(env.obs_space), sep='\n')
    print('Action space:', embodied.format(env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
    timer.wrap('env', env, ['step'])
    timer.wrap('replay', replay, ['add', 'save'])
    timer.wrap('logger', logger, ['write'])

    nonzeros = set()

    import json

    def load_cumul_metrics(directory, config):
        with open(directory / 'metrics.jsonl', "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            last_data = None
            for output in reversed(data):
                if 'episode/total_violations' in output.keys():
                    last_data = output
                    break
            if last_data is None:
                return {'total_violations': 0.0}
            total_violations = float(last_data['episode/total_violations'])
        return {'total_violations': total_violations}

    if (logdir / 'metrics.jsonl').exists():
        cumul_metrics = load_cumul_metrics(logdir, config)
    else:
        cumul_metrics = {'total_violations': 0.0}

    def per_episode(ep, cumul_metrics):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        cost = float(ep['cost'].astype(np.float64).sum())
        violations = cost / config.env.atari.cost_val
        overrides = float(ep['override'].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
        cumul_metrics['total_violations'] += violations
        logger.add({
            'length': length,
            'score': score,
            'cost': cost,
            'violations': violations,
            'overrides': overrides,
            'total_violations': cumul_metrics['total_violations'],
            'sum_abs_reward': sum_abs_reward,
            'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
            'violation_rate': violations / length,
            'override_rate': overrides / length,
        }, prefix='episode')
        print(f'Episode has {length} steps, return {score:.1f} and cost {cost:.1f}.')
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix='stats')

    driver = embodied.Driver(env, cumul_metrics)
    driver.on_episode(lambda ep, worker, cumul_metrics: per_episode(ep, cumul_metrics))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    print('Prefill train dataset.')
    random_agent = embodied.RandomAgent(env.act_space)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]
    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope('dataset'):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix='train')
            if 'priority' in outs:
                replay.prioritize(outs['key'], outs['priority'])
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if 'train/' + k not in agg}
            logger.add(agg)
            logger.add(report, prefix='report')
            logger.add(replay.stats, prefix='replay')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    timer.wrap('checkpoint', checkpoint, ['save', 'load'])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print('Start training loop.')
    policy = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()

    for obj in cleanup:
        obj.close()

if __name__ == '__main__':
    main()
