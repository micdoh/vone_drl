import pytest
import gym
import yaml
import pandas as pd
import stable_baselines3.common.env_checker
from pathlib import Path
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO
from env.envs.VoneEnv import (
    VoneEnv,
    VoneEnvRoutingOnly,
    VoneEnvNodeSelectionOnly,
    VoneEnvNoSorting,
    VoneEnvRoutingMasking
)
from heuristics import *


@pytest.fixture
def setup_vone_env():
    conf = yaml.safe_load(Path("./test/config0.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env


@pytest.fixture
def setup_vone_env_no_node_sorting():
    conf = yaml.safe_load(Path("./test/config_no_node_sorting.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env


@pytest.fixture
def setup_vone_env_node_only():
    conf = yaml.safe_load(Path("./test/config0_nodes_fdl.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env


@pytest.fixture
def setup_vone_env_routing_masking():
    conf = yaml.safe_load(Path("./test/config_routing_masking.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env


def test_check_env():
    stable_baselines3.common.env_checker.check_env(VoneEnv)


def test_vone_env(setup_vone_env):
    env = setup_vone_env
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        action_mask = env.valid_action_mask()
    assert 1 == 1


def test_select_random_action_index(setup_vone_env):
    env = setup_vone_env
    action = select_random_action(env, action_index=1)
    assert isinstance(int(action), int)


def test_select_random_action(setup_vone_env):
    env = setup_vone_env
    action = select_random_action(env)
    assert isinstance(action, np.ndarray)


def test_vone_env_nodes_fdl(setup_vone_env_node_only):
    env = setup_vone_env_node_only
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        action_mask = env.action_masks()
    assert 1 == 1


def test_vone_env_no_node_sorting(setup_vone_env_no_node_sorting):
    env = setup_vone_env_no_node_sorting
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        action_mask = env.action_masks()
    assert 1 == 1


def test_vone_env_routing_masking(setup_vone_env_routing_masking):
    env = setup_vone_env_routing_masking
    obs = env.reset()
    model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, seed=32, verbose=1)
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, done, info = env.step(action)
        print(info)
    assert 1 == 1
