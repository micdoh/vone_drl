"""See https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html"""
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, data_file, model_file, env=None, verbose=0, save_every=1000, save_model=False, save_episode_info=True):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        self.env = env  # type: Union[gym.Env, VecEnv, None]
        self.data_file = Path(data_file) if isinstance(data_file, str) else data_file
        self.model_file = Path(model_file) if isinstance(model_file, str) else model_file
        self.step_count = 0
        self.total_step_count = 0
        self.data = []
        self.save_every = save_every
        self.total_episode_reward = 0
        self.record_episode_reward = -1e8
        self.save_model = save_model
        self.episode_length = env.get_attr("episode_length")[0]
        self.save_episode_info=save_episode_info
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _init_callback(self) -> None:
        # Create folder if needed
        self.data_file.parent.mkdir(exist_ok=True)
        self.model_file.parent.mkdir(exist_ok=True)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        logger.info("Starting new rollout")
        pass

    def _on_step(self) -> None:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered

        :return: (bool) If the callback returns False, training is aborted early.
        """
        current_info = self.env.get_attr("current_info")
        for info in current_info:
            self.data.append(info)
            self.step_count += 1
            self.total_step_count += 1
            self.total_episode_reward += info["reward"]

            if self.step_count >= self.episode_length*len(current_info):

                if self.save_episode_info:
                    df = pd.DataFrame(self.data)
                    df.to_csv(self.data_file, mode='a', header=not os.path.exists(self.data_file))
                    logger.info(f"Appending callback data to {self.data_file.resolve()}")

                self.data = []
                self.step_count = 0

                logger.warning(
                    f"No. timesteps: {self.total_step_count} \n"
                    f"Best mean reward: {self.record_episode_reward/self.episode_length:.2f} \n"
                    f"Last mean reward per episode: {self.total_episode_reward/self.episode_length:.2f}"
                )
                if self.total_episode_reward > self.record_episode_reward:

                    self.record_episode_reward = self.total_episode_reward

                    if self.save_model:
                        logger.warning(
                            "Saving new best model to {}".format(self.model_file.resolve())
                        )
                        self.model.save(self.model_file)

                self.total_episode_reward = 0


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
