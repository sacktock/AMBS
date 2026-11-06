"""Import surface for the pinned scientific stack and the AMBS package.

These stay independent of environment construction so a broken dependency
install fails here rather than inside a training launch.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "numpy",
    "jax",
    "jaxlib",
    "tensorflow",
    "tensorflow_probability",
    "tf_keras",
    "optax",
    "chex",
    "gymnasium",
    "gym",
    "cv2",
    "ruamel.yaml",
    "ambs.embodied",
    "ambs.common.ninjax",
    "ambs.prescience",
    "ambs.model_checking.dfa",
    "ambs.model_checking.properties.property_1",
    "ambs.model_checking.properties.property_3",
    "ambs.embodied.envs.atari",
    "ambs.embodied.envs.seaquest",
    "ambs.dreamerv3.agent",
    "ambs.dreamerv3_lag.agent",
    "ambs.dreamerv3_shield.agent",
    "ambs.cli",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_module_imports(module_name):
    importlib.import_module(module_name)
