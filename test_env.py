import pytest
import gym
import yaml
from pathlib import Path
from sb3_contrib.common.wrappers import ActionMasker
from env.envs.VoneEnv import VoneEnv, VoneEnvRoutingOnly, VoneEnvNodeSelectionOnly
from heuristics import *


@pytest.fixture
def setup_vone_env():
    conf = yaml.safe_load(Path("./test/config0.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"]
    )
    return env


def test_vone_env(setup_vone_env):
    env = setup_vone_env
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    assert 1 == 1