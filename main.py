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
from util_funcs import mask_fn, make_env, linear_schedule
from env.envs.VoneEnv import VoneEnv, VoneEnvRoutingOnly, VoneEnvNodeSelectionOnly


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
        "-t",
        "--test",
        action="store_true",
        help="Disable additional features e.g. weights and biases logging",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,#0.0004681826280349342,
        type=float,
        help="Learning rate for optimisation",
    )
    parser.add_argument(
        "--gamma", default=0.6131640958222716, type=float, help="Discount factor"
    )
    parser.add_argument(
        "--gae_lambda",
        default=0.908,
        type=float,
        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator",
    )
    parser.add_argument(
        "--n_steps",
        default=47,
        type=int,
        help="The number of steps to run for each environment per update "
        "(i.e. rollout buffer size is n_steps * n_envs where n_envs "
        "is number of environment copies running in parallel)",
    )
    parser.add_argument(
        "--linear_scheduler",
        default=0,
        type=int,
        help="Use linear schedule to decay learning rate",
    )
    parser.add_argument(
        "--batch_size", default=64, type=str, help="No. of samples per batch"
    )
    parser.add_argument(
        "--n_procs",
        default=1,
        type=int,
        help="No. of processes to run parallel environments",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Bool to use parallel processes with vectorised environments",
    )
    parser.add_argument("--log", default="WARN", type=str, help="Set log level")
    parser.add_argument(
        "--masking", action="store_true", help="Use invalid action masking"
    )
    parser.add_argument(
        "--model_file", default="", type=str, help="Path to saved model zip file"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Use callback to save model with best training reward",
    )
    args = parser.parse_args()
    conf = yaml.safe_load(Path(args.file).read_text())
    callbacks = []

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    # Setup wandb run
    if not args.test:
        wandb.setup(wandb.Settings(program="main.py", program_relpath="main.py"))
        run = wandb.init(
            project=conf["project"],
            config=conf["wandb_config"],
            dir=conf["log_dir"],
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        (
            log_dir,
            log_file,
            logger,
            model_save_file,
            tensorboard_log_file,
            monitor_file,
        ) = define_paths(run.id, conf, args.log)
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            )
        )
        wandb.define_metric("episode_number")
        wandb.define_metric("acceptance_ratio", step_metric="episode_number")
        conf["env_args"]["wandb_log"] = True
    else:
        (
            log_dir,
            log_file,
            logger,
            model_save_file,
            tensorboard_log_file,
            monitor_file,
        ) = define_paths(start_time, conf, args.log)

    # Setup environment
    env = [
        make_env(conf["env_name"], seed=i, **conf["env_args"])
        for i in range(args.n_procs)
    ]
    env = (
        SubprocVecEnv(env, start_method="fork")
        if args.multithread
        else DummyVecEnv(env)
    )

    # Define callbacks
    callbacks.append(
        CustomCallback(
            env=env,
            data_file=conf["data_file"],
            model_file=conf["model_file"],
            save_model=args.save_model,
        )
    )

    callback_list = CallbackList(callbacks)

    # Create agent
    agent_kwargs = dict(
        verbose=0,
        device="cuda",
        gamma=args.gamma,
        learning_rate=args.learning_rate if not args.linear_scheduler else linear_schedule(args.learning_rate),
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        tensorboard_log=tensorboard_log_file.resolve(),
    )
    agent_args = ("MultiInputPolicy", env)
    model = (
        MaskablePPO(*agent_args, **agent_kwargs)
        if args.masking
        else PPO(*agent_args, **agent_kwargs)
    )

    model.learn(
        total_timesteps=conf["wandb_config"]["total_timesteps"], callback=callback_list
    )

    env.close()
    if not args.test:
        run.finish()
