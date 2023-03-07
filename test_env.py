import pytest
import gymnasium as gym
import yaml
import pandas as pd
import stable_baselines3.common.env_checker
from pathlib import Path
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
from env.envs.VoneEnv import (
    VoneEnv, VoneEnvNodes, VoneEnvPaths
)
from heuristics import *
from util_funcs import *


@pytest.fixture
def setup_vone_env():
    conf = yaml.safe_load(Path("./test/config0.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env

@pytest.fixture
def setup_vone_env_4node():
    conf = yaml.safe_load(Path("./test/4node.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env

def setup_multistep_masking_agent(env):
    model = MaskablePPO("MultiInputPolicy",
                        env,
                        gamma=0.4,
                        seed=1,
                        verbose=1,
                        multistep_masking=True,
                        multistep_masking_attr="curr_selection",
                        multistep_masking_n_steps=3,
                        action_interpreter="select_nodes_paths_slots",
                        )
    return model


@pytest.fixture
def setup_vone_env_no_node_sorting():
    conf = yaml.safe_load(Path("./test/config_no_node_sorting.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    return env


@pytest.fixture
def setup_env_agent_masked():
    conf = yaml.safe_load(Path("./test/config_agent_masked.yaml").read_text())
    env = gym.make(
        conf["env_name"], **conf["env_args"], seed=1
    )
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
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


def test_vone_env_4node_multistep_mask(setup_vone_env_4node):
    env = setup_vone_env_4node
    obs = env.reset()
    model = setup_multistep_masking_agent(env)
    n_steps = 100
    info_list = []
    for _ in range(n_steps):
        # Random action
        action, _states = model.predict(obs, env=env)
        obs, reward, done, info = env.step(action)
        if reward == 0:
            max_vnet_size = env.envs[0].max_vnet_size
            path_action_dim = env.envs[0].k_paths * env.envs[0].num_selectable_slots
            path_mask = env.envs[0].path_mask
            node_mask = env.envs[0].node_mask
            # check if final path_action_dim values of path mask are 0:
            path_mask_selection_sums = [np.sum(path_mask[-path_action_dim*n:-path_action_dim*(n-1)]) for n in range(1, max_vnet_size+1)]
            node_mask_sum = np.sum(node_mask)
            if not ((0 in path_mask_selection_sums) or (node_mask_sum == 0)):
                assert 0 == 1
        info_list.append(info)
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
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, done, info = env.step(action)
        print(info)
    assert 1 == 1

def test_env_agent_masked(setup_env_agent_masked):
    env = setup_env_agent_masked
    #selection = getattr(env, "select_nodes_paths_slots")(1,2,3)
    selection = env.get_attr("select_nodes_paths_slots")[0]([1,2,3])
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        action_mask = env.action_masks()
    assert 1 == 1