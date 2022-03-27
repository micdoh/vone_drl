import gym
import yaml
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from log_init import init_logger
import os
from gym.envs.registration import register
import wandb
import argparse
import numpy as np
from datetime import datetime
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    # Don't forget to use MaskablePPO and ActionMasker wrapper for the env.
    return env.valid_action_mask()


def define_paths(run_id, conf):
    log_dir = Path(conf["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    logger = init_logger(log_file.absolute(), log_file.stem)
    logger.info(f"Config file {args.file} contents:\n {conf}")
    model_save_file = log_dir / f"{run_id}_model.zip"
    tensorboard_log_file = log_dir / f"{run_id}_tensorboard.log"
    monitor_file = log_dir / f"{run_id}_monitor.csv"
    return log_dir, log_file, logger, model_save_file, tensorboard_log_file, monitor_file


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='Absolute path to config file for run')
parser.add_argument('-t', '--test', action='store_true', help='Disable additional features e.g. weights and biases logging')
args = parser.parse_args()
conf = yaml.safe_load(Path(args.file).read_text())

start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

register(
    id='vone_Env-v0',
    entry_point='env.envs:VoneEnv',
)

callbacks = [
    SaveOnBestTrainingRewardCallback(
        check_freq=5000,
        log_dir=conf["log_dir"]
    )
]

log_dir, log_file, logger, model_save_file, tensorboard_log_file, monitor_file = define_paths(start_time, conf)

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
    log_dir, log_file, logger, model_save_file, tensorboard_log_file, monitor_file = define_paths(run.id, conf)
    callbacks.append(WandbCallback(
        gradient_save_freq=0,
        model_save_path=model_save_file.resolve(),
        verbose=2,
    ))
    wandb.define_metric("episode_number")
    wandb.define_metric("acceptance_ratio", step_metric="episode_number")
    conf['env_args']['wandb_log'] = True

callback_list = CallbackList(callbacks)

info_keywords = ('P_accepted', 'topology_name', 'load',)
env = gym.make(conf["env_name"], **conf["env_args"])
env = Monitor(env, filename=str(monitor_file.resolve()), info_keywords=info_keywords)

# create agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    device='cuda',
    gamma=0.6,
    n_steps=10,
    tensorboard_log=tensorboard_log_file.resolve()
)

# TODO - Add training hyperparameters to config
model.learn(total_timesteps=200000, callback=callback_list)

eva = evaluate_policy(model, env, n_eval_episodes=10)
env.close()
if not args.test:
    run.finish()
