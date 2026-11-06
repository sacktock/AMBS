"""Launch one debug training run on CPU.

The script calls the selected algorithm's ``train.main`` with the debug
configuration. Shield runs keep the debug ``shield_samples`` and set
``expl_until`` so the training-loop policy steps call the shield.

Equivalent console command for an Assault shield smoke::

    ambs train dreamerv3-shield --configs atari debug --task atari_assault \\
        --env.atari.labels overheat death --shielding_type safe_prod_sum \\
        --envs.amount 1 --envs.parallel none --jax.platform cpu \\
        --run.expl_until 1000000 --run.steps 200 --logdir logdir/smoke/assault
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

ALGORITHMS = {
    "dreamerv3": "ambs.dreamerv3.train",
    "dreamerv3-lag": "ambs.dreamerv3_lag.train",
    "dreamerv3-shield": "ambs.dreamerv3_shield.train",
}
ENVS = (
    "pointgoal1",
    "assault",
    "seaquest_property_1",
    "seaquest_property_3",
)
SHIELDING_TYPES = (
    "task_prod",
    "safe_prod_sum",
    "safe_prod_max",
    "safe_prod_cond",
)


def build_argv(algorithm: str, env: str, shielding_type: str | None, logdir: Path) -> list[str]:
    if algorithm == "dreamerv3-shield" and shielding_type is None:
        raise SystemExit("dreamerv3-shield requires --shielding-type")
    if algorithm != "dreamerv3-shield" and shielding_type is not None:
        raise SystemExit("--shielding-type is only used with dreamerv3-shield")
    if env == "pointgoal1":
        argv = [
            "--configs", "safetygym", "debug",
            "--task", "safetygym_Safexp-PointGoal1-v0",
        ]
    elif env == "assault":
        argv = [
            "--configs", "atari", "debug",
            "--task", "atari_assault",
            "--env.atari.labels", "overheat", "death",
        ]
    elif env == "seaquest_property_1":
        argv = [
            "--configs", "seaquest", "debug",
            "--env.seaquest.property_file", "property_1",
        ]
    elif env == "seaquest_property_3":
        argv = [
            "--configs", "seaquest", "debug",
            "--env.seaquest.property_file", "property_3",
        ]
    else:
        raise SystemExit(f"Unknown environment {env}")
    argv += [
        "--envs.amount", "1",
        "--envs.parallel", "none",
        "--jax.platform", "cpu",
        "--run.steps", "200",
        "--logdir", str(logdir),
    ]
    if shielding_type is not None:
        argv += ["--shielding_type", shielding_type, "--run.expl_until", "1000000"]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--shielding-type", choices=SHIELDING_TYPES)
    parser.add_argument("--logdir", type=Path)
    args = parser.parse_args()
    slug = args.env if args.shielding_type is None else f"{args.env}_{args.shielding_type}"
    logdir = args.logdir or Path("logdir") / "smoke" / f"{args.algorithm}_{slug}"
    argv = build_argv(args.algorithm, args.env, args.shielding_type, logdir)
    importlib.import_module(ALGORITHMS[args.algorithm]).main(argv)


if __name__ == "__main__":
    main()
