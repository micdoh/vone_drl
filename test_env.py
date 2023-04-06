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
import env.envs.VoneEnv as VoneEnv
from heuristics import *
from util_funcs import *


@pytest.fixture
def setup_vone_env():
    conf = yaml.safe_load(Path("./test/config0.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env


@pytest.fixture
def setup_vone_env_100_slots():
    conf = yaml.safe_load(Path("./test/agent_combined_100_slots.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = SubprocVecEnv(env, start_method="fork")
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


@pytest.fixture
def setup_vone_env_4node_vector():
    conf = yaml.safe_load(Path("./test/4node.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = SubprocVecEnv(env)
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
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env


@pytest.fixture
def setup_env_agent_masked():
    conf = yaml.safe_load(Path("./test/config_agent_masked.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env


@pytest.fixture
def setup_vone_env_node_only():
    conf = yaml.safe_load(Path("./test/config0_nodes_fdl.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env


@pytest.fixture
def setup_vone_env_routing_masking():
    conf = yaml.safe_load(Path("./test/config_routing_masking.yaml").read_text())
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(1)
    ]
    env = DummyVecEnv(env)
    return env


def get_env_attrs(env):
    if isinstance(env, SubprocVecEnv):
        max_vnet_size = env.get_attr("max_vnet_size")[0]
        path_action_dim = env.get_attr("k_paths")[0] * env.get_attr("num_selectable_slots")[0]
        path_mask = env.get_attr("path_mask")[0]
        node_mask = env.get_attr("node_mask")[0]
    else:
        max_vnet_size = env.envs[0].max_vnet_size
        path_action_dim = env.envs[0].k_paths * env.envs[0].num_selectable_slots
        path_mask = env.envs[0].path_mask
        node_mask = env.envs[0].node_mask
    return max_vnet_size, path_action_dim, path_mask, node_mask

@timeit
def loop_episodes_to_check_masking(env, model, n_steps, reject_action=False):
    info_list = []
    obs = env.reset()
    for _ in range(n_steps):
        # Random action
        action, _states = model.predict(obs, env=env)
        obs, reward, done, info = env.step(action)
        if reward == 0:
            max_vnet_size, path_action_dim, path_mask, node_mask = get_env_attrs(env)
            # check if final path_action_dim values of path mask are 0:
            path_mask_selection_sums = [np.sum(path_mask[path_action_dim*n:path_action_dim*(n+1)]) for n in range(max_vnet_size)]
            node_mask_sum = np.sum(node_mask)
            if reject_action:
                if not action[0][0] == 0:  # Check if reject action selected
                    if not 0 in path_mask_selection_sums:
                        return False  # Fail
            else:
                # If no part path mask is entirely 0 or node mask is 0, then fail
                if not ((0 in path_mask_selection_sums) or (node_mask_sum == 0)):
                    return False  # Fail
        info_list.append(info)
    for item in info_list:
        print(item)
    return True  # Pass


def test_check_env():
    stable_baselines3.common.env_checker.check_env(VoneEnv)


def test_vone_env(setup_vone_env):
    env = setup_vone_env
    model = setup_multistep_masking_agent(env)
    n_steps = 100
    assert loop_episodes_to_check_masking(env, model, n_steps) is True


@pytest.mark.slow
def test_vone_env_100_slots(setup_vone_env_100_slots):
    env = setup_vone_env_100_slots
    model = setup_multistep_masking_agent(env)
    n_steps = 10000
    assert loop_episodes_to_check_masking(env, model, n_steps) is True


def test_vone_env_4node_multistep_mask(setup_vone_env_4node):
    env = setup_vone_env_4node
    model = setup_multistep_masking_agent(env)
    n_steps = 10000
    assert loop_episodes_to_check_masking(env, model, n_steps) is True


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