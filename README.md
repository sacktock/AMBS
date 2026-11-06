# Approximate Model-Based Shielding (AMBS)

World-model look-ahead for safe reinforcement learning. DreamerV3 rolls out candidate futures in latent space, estimates the chance of a safety violation, and overrides the task policy with a safe policy when that chance is too high.

Papers:

- [Approximate Model-Based Shielding for Safe Reinforcement Learning](https://arxiv.org/abs/2308.00707)
- [Leveraging Approximate Model-based Shielding for Probabilistic Safety Guarantees in Continuous Environments](http://arxiv.org/abs/2402.00816)

The arXiv versions include the supplementary material.

## Software stack

Training runs on CPU or CUDA with [JAX](https://docs.jax.dev/) 0.4.30. [TensorFlow](https://www.tensorflow.org/) 2.17, `tf-keras` 2.17, and [TensorFlow Probability](https://www.tensorflow.org/probability) 0.24 supply action distributions and TensorBoard. Atari and Seaquest use [Gymnasium](https://gymnasium.farama.org/) and ALE. Safety Gym still uses classic Gym and is installed separately.

Python 3.12 is required.

## Installation

### conda

- Install conda, for example from [Anaconda](https://anaconda.org/channels/anaconda/packages/conda/overview).
- Clone the repo:

```bash
git clone https://github.com/sacktock/AMBS.git
cd AMBS
```

- Create and activate the environment:

```bash
conda env create --name ambs --file conda-environment.yaml
conda activate ambs
```

- Install the project and its dependencies:

```bash
pip install -e .
```

`pip install -e .` installs the dependencies declared in `pyproject.toml`. `requirements.txt` lists the same base dependencies and is kept in lockstep with that file. Chex 0.1.87 is part of that set.

### uv

From the repository root, `uv sync` installs the base dependency set (CPU JAX):

```bash
uv sync
```

## Enabling GPU acceleration with JAX (optional)

CPU is enough to smoke-test the launcher. GPU acceleration is optional.

**macOS and Windows:** JAX with CPU is the recommended install. No further action is required after the steps above.

#### pip

**Linux x86_64/aarch64** and **Windows WSL2 x86_64**: jax and jaxlib `0.4.30` are already installed. Reinstall JAX for the CUDA driver. Do not use the `-U` option.

```bash
pip install "jax[cuda12]==0.4.30"
```

CUDA 13 does not ship with this stack. JAX 0.4.30 only publishes a CUDA 12 extra.

#### uv

```bash
uv sync --group cuda12
```

## Safety Gym

Safety Gym experiments need [MuJoCo](http://mujoco.org/) 2.1 and the Safety Gym fork. These are not project dependencies.

### Download MuJoCo binaries

1. Download the MuJoCo 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

### Install Safety Gym

```bash
git clone https://github.com/sacktock/safety-gym
```

First make sure `setup.py` has the following dependency list:

```python
install_requires=[
        'gym>=0.15.3',
        'joblib>=0.14.0',
        'Cython==0.29.36',
        'mujoco-py @ git+https://github.com/sacktock/mujoco-py.git@master',
        'numpy>=1.17.4',
        'xmltodict>=0.12.0',
    ],
```

Then:

```bash
cd safety-gym
python -m pip install -e .
```

Moving Safety Gym onto [safety-gymnasium](https://github.com/Farama-Foundation/Safety-Gymnasium) is a future plan. The current environments stay on Safety Gym so existing results can be reproduced.

### In-cluster extra

```bash
conda install -c conda-forge glew
```

**HPC only:**

```bash
conda install -c conda-forge "mesalib>=25.0.5,<25.1.0"
```

```bash
conda install -c conda-forge patchelf
```

*Optional (on HPC, `module load libglvnd` is an alternative):*

```bash
conda install -c conda-forge libgl-devel
```

## How to run

After `pip install -e .`, the `ambs` console script is on `PATH`. With uv, pass the same arguments to `uv run ambs`.

```bash
ambs train dreamerv3 --configs atari debug --task atari_assault --env.atari.labels overheat death --jax.platform cpu --envs.amount 1 --envs.parallel none --run.steps 200 --logdir logdir/assault/base
```

```bash
uv run ambs train dreamerv3 --configs atari debug --task atari_assault --env.atari.labels overheat death --jax.platform cpu --envs.amount 1 --envs.parallel none --run.steps 200 --logdir logdir/assault/base
```

`train` selects the algorithm. Everything after the algorithm name is parsed by that algorithm's `train.py`, the same flags as before.

| Command | Algorithm |
| --- | --- |
| `ambs train dreamerv3` | DreamerV3 |
| `ambs train dreamerv3-lag` | Lagrangian DreamerV3 |
| `ambs train dreamerv3-shield` | Shielded DreamerV3 |

`--configs` takes one or more blocks from that algorithm's `configs.yaml`. Later blocks override earlier ones. Remaining flags override individual keys (`--seed 0`, `--run.steps 10000000`, `--env.atari.labels death early-surface out-of-oxygen`).

Suites:

- Atari: `--task atari_assault`, `--task atari_seaquest`, and the other names in the Atari labeller map. Labels come from `--env.atari.labels`.
- Safety Gym: `--task safetygym_Safexp-PointGoal1-v0` (and the other Safety Gym ids). `--configs safetygym_vision` turns on image observations.
- Seaquest automata: `--task seaquest` with `--env.seaquest.property_file property_1` or `property_3`.

Shielded runs take `--shielding_type` of `task_prod`, `safe_prod_sum`, `safe_prod_max`, or `safe_prod_cond`. Penalty training still uses `--penalty_coeff`, `--plpg`, and `--copt`.

DreamerV3 with shielding on Seaquest for the Atari labels, 10M environment steps:

```bash
ambs train dreamerv3-shield \
  --logdir ./logdir/seaquest/shield \
  --configs atari xlarge \
  --task atari_seaquest \
  --env.atari.labels death early-surface out-of-oxygen \
  --shielding_type task_prod \
  --run.steps 10000000
```

Seaquest property 3 with a safe-product shield:

```bash
ambs train dreamerv3-shield \
  --logdir ./logdir/seaquest/safe_prod_sum \
  --configs seaquest medium \
  --shielding_type safe_prod_sum \
  --env.seaquest.property_file property_3 \
  --run.steps 5000000
```

PointGoal1 with the penalty critic:

```bash
ambs train dreamerv3-shield \
  --logdir ./logdir/safetygym/PointGoal1/shield_penl \
  --configs safetygym_vision large \
  --task safetygym_Safexp-PointGoal1-v0 \
  --penalty_coeff 1.0 \
  --normalise_ret False \
  --penl_critic_type vfunction \
  --run.steps 500000
```

Probabilistic logic policy gradient:

```bash
ambs train dreamerv3-shield \
  --logdir ./logdir/safetygym/PointGoal1/shield_plpg \
  --configs safetygym_vision large \
  --task safetygym_Safexp-PointGoal1-v0 \
  --penalty_coeff 0.8 \
  --plpg True \
  --normalise_ret False \
  --penl_critic_type vfunction \
  --run.steps 500000
```

Counter-example guided policy optimisation:

```bash
ambs train dreamerv3-shield \
  --logdir ./logdir/safetygym/PointGoal1/shield_copt \
  --configs safetygym_vision large \
  --task safetygym_Safexp-PointGoal1-v0 \
  --penalty_coeff 1.0 \
  --copt True \
  --normalise_ret False \
  --penl_critic_type vfunction \
  --run.steps 500000
```

`examples/smoke.py` launches one debug CPU run (a short prefill plus one training chunk) for a chosen algorithm, environment, and, for the shield, shielding type:

```bash
python examples/smoke.py --algorithm dreamerv3-shield --env assault --shielding-type safe_prod_sum
```

TensorBoard reads the run directory:

```bash
tensorboard --logdir ./logdir/seaquest
```

## Acks

- [DreamerV3](https://github.com/danijar/dreamerv3)
- [Bounded Prescience](https://github.com/HjalmarWijk/bounded-prescience) (safety labels for Atari)
- [LAMBDA](https://github.com/yardenas/la-mbda) (Lagrangian CPO implementation for DreamerV2)
