import gym
import yaml
import wandb
import argparse
import numpy as np
import os
import random
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from typing import Callable
from datetime import datetime
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from callback import SaveOnBestTrainingRewardCallback, CustomCallback
from log_init import init_logger
from heuristics import nsc_ksp_fdl
from util_funcs import mask_fn, make_env
from env.envs.VoneEnv import (
    VoneEnvUnsortedSeparate,
    VoneEnvSortedSeparate,
    VoneEnvNodesSorted,
    VoneEnvNodesUnsorted,
    VoneEnvRoutingSeparate,
    VoneEnvRoutingCombined,
)
import time


def define_paths(run_id, conf, loglevel):
    log_dir = Path(conf["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    logger = init_logger(log_file.absolute(), loglevel)
    logger.info(f"Config file {args.file} contents:\n {conf}")
    model_save_file = log_dir / f"{run_id}_model.zip"
    tensorboard_log_file = log_dir / f"{run_id}_tensorboard.log"
    monitor_file = log_dir / f"{run_id}_monitor.csv"
    return (
        log_dir,
        log_file,
        logger,
        model_save_file,
        tensorboard_log_file,
        monitor_file,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, help="Absolute path to config file for run"
    )
    parser.add_argument(
        "--n_procs",
        default=1,
        type=int,
        help="No. of processes to run parallel environments",
    )
    parser.add_argument(
        "--min_load",
        default=1,
        type=int,
        help="Min. of range of loads to evaluate",
    )
    parser.add_argument(
        "--max_load",
        default=10,
        type=int,
        help="Max. of range of loads to evaluate",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Bool to use parallel processes with vectorised environments",
    )
    parser.add_argument("--log", default="INFO", type=str, help="Set log level")
    parser.add_argument(
        "--masking", action="store_true", help="Use invalid action masking"
    )
    parser.add_argument(
        "--model_file", default="", type=str, help="Path to saved model zip file"
    )
    args = parser.parse_args()
    conf = yaml.safe_load(Path(args.file).read_text())

    (
        log_dir,
        log_file,
        logger,
        model_save_file,
        tensorboard_log_file,
        monitor_file,
    ) = define_paths(0, conf, args.log)

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    results = []
    for load in range(args.min_load, args.max_load + 1):
        callbacks = []
        conf["env_args"]["load"] = load
        env = gym.make(conf["env_name"], seed=load, **conf["env_args"])

        model = MaskablePPO.load(args.model_file) if args.masking else PPO.load(args.model_file)
        #eva = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
        obs = env.reset()
        for _ in range(env.episode_length):

            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)

            results.append(info)

    print(results)
