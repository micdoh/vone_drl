import yaml
import wandb
import pandas as pd
import os
import sys
from stable_baselines3 import PPO
sys.path.append(os.path.abspath("sb3-contrib"))
print(sys.path)
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from datetime import datetime
from stable_baselines3.common.callbacks import CallbackList
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

    output_file = Path(args.output_file).parent / start_time / Path(args.output_file).name
    episode_data_file = Path(args.output_file).parent / start_time / "timestep_data.csv"
    output_file.parent.mkdir(exist_ok=True)

    episode_length = conf["env_args"]["episode_length"]

    if args.artifact:
        logger.warn(f"Downloading artifact {args.artifact}")
        api = wandb.Api()
        artifact = api.artifact(args.artifact)
        model_file = artifact.download(root=Path(__file__).parent / "eval" / "models") / f"{artifact.name.split(':')[0]}.zip"
        print(model_file.resolve())
    else:
        model_file = args.model_file

    results = []
    for load in range(args.min_load, args.max_load + 1, args.load_step):
        callbacks = []
        conf["env_args"]["load"] = load
        env = [make_env(conf["env_name"], seed=load, **conf["env_args"])]
        env = (DummyVecEnv(env))
        agent_args = ("MultiInputPolicy", env)
        agent_kwargs = dict(multistep_masking=args.multistep_masking,
                            multistep_masking_attr=args.multistep_masking_attr,
                            action_interpreter=args.action_interpreter,
                            multistep_masking_n_steps=args.multistep_masking_n_steps,
                            )
        model = (
            MaskablePPO(*agent_args, **agent_kwargs)
            if args.masking
            else PPO(*agent_args, **agent_kwargs)
        )
        model.set_parameters(model_file)
        callback = CustomCallback(
                env=env,
                data_file=episode_data_file,
                model_file=model_save_file,
                save_model=False,
                save_episode_info=True
            ) if args.callback else None
        eva = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_eval_episodes,
            use_masking=args.eval_masking,
            use_multistep_masking=args.eval_multistep_masking,
            callback=callback,
        )
        # This assumes reward is 0 or +10 for every step
        results.append({
            "load": load,
            "reward": eva[0]/episode_length,
            "std": eva[1]/episode_length,
            "blocking": (10-(eva[0]/episode_length))/10,
            "blocking_std": (eva[1]/episode_length)/10,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))
    print(results)
