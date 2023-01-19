import gym
import yaml
import wandb
import argparse
import numpy as np
import pandas as pd
import os
import random
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from typing import Callable
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
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
    return (
        log_dir,
        log_file,
        logger,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, help="Absolute path to config file for run"
    )
    parser.add_argument(
        "--min_load",
        default=1,
        type=int,
        help="Min. of range of loads to evaluate",
    )
    parser.add_argument(
        "--max_load",
        default=15,
        type=int,
        help="Max. of range of loads to evaluate",
    )
    parser.add_argument(
        "--log", default="INFO", type=str, help="Set log level"
    )
    parser.add_argument(
        "--masking", action="store_true", help="Use invalid action masking for model type"
    )
    parser.add_argument(
        "--eval_masking", action="store_true", help="Use invalid action masking for evaluation"
    )
    parser.add_argument(
        "--model_file", default="", type=str, help="Path to load model zip file"
    )
    parser.add_argument(
        "--output_file", default="", type=str, help="Path to output csv file"
    )
    parser.add_argument(
        "--artifact", default=None, type=str, help="Name of artiact to download from wandb"
    )
    parser.add_argument(
        "--recurrent_masking", action="store_true", help="Use recurrent invalid action masking"
    )
    args = parser.parse_args()
    conf = yaml.safe_load(Path(args.file).read_text())

    (
        log_dir,
        log_file,
        logger,
    ) = define_paths(0, conf, args.log)

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    episode_length = conf["env_args"]["episode_length"]

    if args.artifact is not None:
        logger.warn(f"Downloading artifact {args.artifact}")
        api = wandb.Api()
        artifact = api.artifact(args.artifact)
        model_file = artifact.download(root=Path(args.output_file).parent) / f"{artifact.name.split(':')[0]}.zip"
    else:
        model_file = args.model_file

    results = []
    for load in range(args.min_load, args.max_load + 1):
        callbacks = []
        conf["env_args"]["load"] = load
        #env = gym.make(conf["env_name"], seed=load, **conf["env_args"])
        env = [make_env(conf["env_name"], seed=load, **conf["env_args"])]
        env = (DummyVecEnv(env))
        agent_args = ("MultiInputPolicy", env)
        agent_kwargs = dict(recurrent_masking=args.recurrent_masking)
        model = (
            MaskablePPO(*agent_args, **agent_kwargs)
            if args.masking
            else PPO(*agent_args, **agent_kwargs)
        )
        model.set_parameters(model_file)
        #model = MaskablePPO.load(args.model_file) if args.masking else PPO.load(args.model_file)
        eva = evaluate_policy(model, env, n_eval_episodes=3, use_masking=args.eval_masking)
        results.append({
            "load": load,
            "reward": eva[0]/episode_length,
            "std": eva[1]/episode_length,
            "blocking": (1-eva[0]/10)/episode_length,
            "blocking_std": (eva[1]/10)/episode_length,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, mode='a', header=not os.path.exists(args.output_file))
    print(results)
