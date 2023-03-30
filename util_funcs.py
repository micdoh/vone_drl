import gymnasium as gym
import numpy as np
import os
from pathlib import Path
from itertools import islice
from typing import Generator, Any, Callable
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--env_file", type=str, help="Absolute path to config file for run"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable weights and biases logging",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for optimisation",
    )
    parser.add_argument(
        "--gamma", default=0.6, type=float, help="Discount factor"
    )
    parser.add_argument(
        "--clip_range",
        default=0.2,
        type=float,
        help="Clipping range for PPO",
    )
    parser.add_argument(
        "--clip_range_vf",
        default=None,
        type=float,
        help="Clipping range for value function",
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
        "--schedule",
        default=None,
        type=str,
        help="Type of learning rate schedule to use",
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
        "--multistep_masking", action="store_true", help="Use multistep invalid action masking"
    )
    parser.add_argument(
        "--multistep_masking_attr",
        default="curr_selection",
        type=str,
        help="Specify environment variables to be accessed sequentially in multistep masking",
    )
    parser.add_argument(
        "--multistep_masking_n_steps",
        default=3,
        type=int,
        help="Specify number of steps to mask in multistep masking",
    )
    parser.add_argument(
        "--action_interpreter",
        default="select_nodes_paths_slots",
        help="Specify function that interprets actions in the env. To be used in multistep action masking.",
    )
    parser.add_argument(
        "--eval_masking", action="store_true", help="Eval: Use invalid action masking for evaluation"
    )
    parser.add_argument(
        "--output_file", default="", type=str, help="Eval: Path to output csv file"
    )
    parser.add_argument(
        "--min_load",
        default=1,
        type=int,
        help="Eval: Min. of range of loads to evaluate",
    )
    parser.add_argument(
        "--max_load",
        default=15,
        type=int,
        help="Eval: Max. of range of loads to evaluate",
    )
    parser.add_argument(
        "--model_file", default="", type=str, help="Path to model zip file for retraining"
    )
    parser.add_argument(
        "--artifact", default="", type=str, help="Name of artiact to download from wandb"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Use callback to save model with best training reward",
    )
    parser.add_argument(
        "--policy_type",
        default="MultiInputPolicy",
        help="Specify type of policy to be used by agent",
    )
    parser.add_argument(
        "--total_timesteps",
        default=135000,
        type=int,
        help="Specify type of policy to be used my agent",
    )
    parser.add_argument(
        "-i", "--id", type=str, help="ID to name this evaluation run"
    )
    parser.add_argument(
        "--save_episode_info",
        action="store_true",
        help="Save episode rewards to file",
    )
    parser.add_argument(
        "--log_dir",
        default="./runs",
        help="Parent directory for all log directories",
    )

    return parser.parse_args()



def init_logger(log_file: str, loglevel: str):
    logger = logging.getLogger()
    loglevel = getattr(logging, loglevel.upper())
    logger.setLevel(loglevel)
    # create file handler which logs event debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(loglevel)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def choose_schedule(schedule: str, initial_value: float):
    """Choose schedule for learning rate"""
    if schedule is None:
        return initial_value
    elif schedule == "linear":
        return linear_schedule(initial_value)
    else:
        raise ValueError("Schedule not recognised")


def define_logs(run_id, log_dir, loglevel):
    log_dir = Path(log_dir) / run_id
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    logger = init_logger(log_file.absolute(), loglevel)
    model_save_file = log_dir / f"{run_id}_model.zip"
    return (
        log_dir,
        log_file,
        logger,
        model_save_file,
    )


def get_nth_item(gen: Generator, n: int) -> Any:
    """Return the nth item from a generator"""
    return next(islice(gen, n, None), None)


def get_gen_index(gen: Generator, value: [int]) -> int:
    """Get index of generator value that matches value.

    Used to translate from requirement value back to request integer,
    which is the index of the matching table row.

    Args:
        gen: Generator to retrieve value-index from.
        value: list of values to match to generator
    """
    for n, item in enumerate(gen):
        if (item == np.array(value)).all():
            return n


def mask_fn(env: gym.Env) -> np.ndarray:
    """Do whatever you'd like in this function to return the action mask for the env.
    In this example, we assume the env has a helpful method we can rely on.
    Don't forget to use MaskablePPO and ActionMasker wrapper for the env."""
    return env.valid_action_mask()


def make_env(env_id, seed, **kwargs):
    """Used for instantiating multiple (vectorised) envs.
    Disable env checker to ensure env is not wrapped (to set env attrs for action masking)"""
    def _init():
        env = gym.make(env_id, disable_env_checker=True, **kwargs)
        env.seed(seed)
        return env

    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
