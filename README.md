# Approximate Model-based Shielding (AMBS)

This repository provides the latest official implementation of **Approximate Model-based Shielding** including spin-off papers. The general idea is that by leveraging worlds models we can obtain latent risk estimates for safety property violation of reinforcement learning policies. In conjunction with the latent risk estimators a dedicated safe backup policy guides learning through *shielding* -- action replacement -- when the chance of a safety violation in the near future is too high. 

Papers:

- *Seminal paper:* [Approximate Model-Based Shielding for Safe Reinforcement Learning](https://arxiv.org/abs/2308.00707)
- *Extension and evaluation for continuous state-action spaces:* [Leveraging Approximate Model-based Shielding for Probabilistic Safety Guarantees in Continuous Environments](http://arxiv.org/abs/2402.00816)
- *Extension to more expression non-Markovian safety properties*: [RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning](https://arxiv.org/abs/2610.23445)

Supplementary materials (e.g., training details) are provided in the arXiv versions. 

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

## Contributing

### Tests

With uv:

```bash
uv sync --group dev
uv run pytest -v
```

With conda or pip, install the project, then pytest, and run from the repository root:

```bash
pip install -e .
pip install "pytest>=9"
pytest -v
```

`pytest` is the local test run, and it is what the `tests` job runs. It checks that the scientific stack imports and that the packaged algorithms, Atari and Seaquest modules, and model-checking properties load. It does not install Atari ROMs or Safety Gym, and it does not start training.

The full smoke matrix is a separate GitHub Actions job, `smoke`, and it is not part of `pytest`. That job runs on Ubuntu only. It calls `examples/smoke.py` for every algorithm and environment: DreamerV3 and Lagrangian DreamerV3 once per environment, and shielded DreamerV3 once per environment for `task_prod`, `safe_prod_sum`, `safe_prod_max`, and `safe_prod_cond`. Each cell is one debug CPU update. The job installs Atari ROMs for Assault and Seaquest, and MuJoCo 2.1 plus Safety Gym for PointGoal1. Those installs stay out of the local `pytest` run because the matrix is large and pulls in ROMs, MuJoCo, and a training launch per cell.

One cell can still be launched locally:

```bash
# conda or pip
python examples/smoke.py --algorithm dreamerv3-shield --env assault --shielding-type safe_prod_sum

# uv
uv run python examples/smoke.py --algorithm dreamerv3-shield --env assault --shielding-type safe_prod_sum
```

Assault and Seaquest need ROMs first (`uv sync --group smoke`, then `uv run --group smoke AutoROM --accept-license`). PointGoal1 needs the Safety Gym install above.

### Dependency sync

`requirements.txt` must match `[project.dependencies]` in `pyproject.toml`. CI runs `scripts/check_requirements_sync.py` when either file changes. Locally:

```bash
python scripts/check_requirements_sync.py
```

## Acknowledgements

- *Original DreamerV3 repository:* [DreamerV3](https://github.com/danijar/dreamerv3)
- *Safety labels for Atari envs:* [Bounded Prescience](https://github.com/HjalmarWijk/bounded-prescience)
- *Augmented Lagrangian CMDP implementation for DreamerV2:* [LAMBDA](https://github.com/yardenas/la-mbda)

## License

Our implementation of AMBS is released under the MIT License.
