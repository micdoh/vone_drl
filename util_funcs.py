import gym
import numpy as np
from itertools import islice
from typing import Generator, Any, Callable


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
    """Used for instantiating multiple (vectorised) envs"""
    def _init():
        env = gym.make(env_id, **kwargs)
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
