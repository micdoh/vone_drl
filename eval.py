import yaml
import wandb
import pandas as pd
import os
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from callback import CustomCallback
from util_funcs import mask_fn, make_env, define_logs, parse_args
import env.envs.VoneEnv as VoneEnv


if __name__ == "__main__":

    args = parse_args()
    conf = yaml.safe_load(Path(args.env_file).read_text())

    (
        log_dir,
        log_file,
        logger,
        model_save_file
    ) = define_logs(args.id, args.log_dir, args.log)

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    episode_length = conf["env_args"]["episode_length"]

    if args.artifact is not None:
        logger.warn(f"Downloading artifact {args.artifact}")
        api = wandb.Api()
        artifact = api.artifact(args.artifact)
        model_file = artifact.download(root=Path(args.output_file).parent) / f"{artifact.name.split(':')[0]}.zip"
        print(model_file.resolve())
    else:
        model_file = args.model_file

    results = []
    for load in range(args.min_load, args.max_load + 1):
        callbacks = []
        conf["env_args"]["load"] = load
        env = [make_env(conf["env_name"], seed=load, **conf["env_args"])]
        env = (DummyVecEnv(env))
        agent_args = ("MultiInputPolicy", env)
        agent_kwargs = dict(multistep_masking=args.multistep_masking,
                            multistep_masking_terms=[args.multistep_masking_terms],
                            action_interpreter=args.action_interpreter,
                            )
        model = (
            MaskablePPO(*agent_args, **agent_kwargs)
            if args.masking
            else PPO(*agent_args, **agent_kwargs)
        )
        model.set_parameters(model_file)
        eva = evaluate_policy(model, env, n_eval_episodes=3, use_masking=args.eval_masking)
        results.append({
            "load": load,
            "reward": eva[0]/episode_length,
            "std": eva[1]/episode_length,
            "blocking": (10-(eva[0]/episode_length))/10,
            "blocking_std": (eva[1]/episode_length)/10,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, mode='a', header=not os.path.exists(args.output_file))
    print(results)
