import numpy as np
import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from log_init import init_logger
import os
from gym.envs.registration import register

register(
    id='vone_Env-v0',
    entry_point='env.envs:VoneEnv',
)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)

        return True


log_dir = Path("./tmp/vone_Env-v0/")
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / f"training.log"
logger = init_logger(log_file.resolve(), log_file.stem)
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)

env_args = dict(episode_length=5000, load=3, mean_service_holding_time=10, k_paths=2)
the_env = gym.make('vone_Env-v0', **env_args)
the_env = Monitor(the_env, str(log_file.resolve()), info_keywords=('P_accepted', 'topology_num', 'load',))

# create agent
model = A2C("MultiInputPolicy", the_env, verbose=1, device='cuda', gamma=0.6, n_steps=10)  # ,learning_rate=0.0007
model.learn(total_timesteps=200000)

eva = evaluate_policy(model, the_env, n_eval_episodes=10)
the_env.close()

loop = True
if loop is True:
    logs = ["./tmp/network_Env_0/", "./tmp/network_Env_1/", "./tmp/network_Env_2/", "./tmp/network_Env_3/",
            "./tmp/network_Env_4/", "./tmp/network_Env_5/", "./tmp/network_Env_6/", "./tmp/network_Env_7/",
            "./tmp/network_Env_8/", "./tmp/network_Env_9/", "./tmp/network_Env_10/", "./tmp/network_Env_11/"]
    load = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
    for i in range(len(load)):
        os.makedirs(logs[i], exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=logs[i])

        env_args_2 = dict(
            episode_length=5000,
            load=load[i],
            mean_service_holding_time=10,
            k_paths=2,
            topology_num=None
        )
        the_env = gym.make('network_Env-v0', **env_args)
        the_env = Monitor(the_env, logs[i] + 'training', info_keywords=('P_accepted', 'topology_num', 'load',))

        eva = evaluate_policy(model, the_env, n_eval_episodes=5)
        the_env.close()
