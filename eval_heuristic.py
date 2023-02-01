import gym
from heuristics import nsc_ksp_fdl
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import env.envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", default="", type=str, help="Path to output csv file"
    )
    parser.add_argument(
        "--min_load", default=1, type=int, help="Minimum load"
    )
    parser.add_argument(
        "--max_load", default=15, type=int, help="Maximum load"
    )
    parser.add_argument(
        "--num_episodes", default=3, type=int, help="Number of episodes"
    )
    parser.add_argument(
        "--episode_length", default=5000, type=int, help="Episode length"
    )
    parser.add_argument(
        "--k_paths", default=5, type=int, help="Number of paths"
    )
    parser.add_argument(
        "--mean_service_holding_time", default=10, type=int, help="Mean service holding time"
    )
    parser.add_argument(
        "--env_name", default="vone_Env-v0", type=str, help="Environment name"
    )
    args = parser.parse_args()

    data_file = Path(args.output_file)

    data_file.parent.mkdir(exist_ok=True)

    for load in range(args.min_load, args.max_load+1):

        env_args = dict(
            episode_length=args.episode_length,
            load=load,
            mean_service_holding_time=args.mean_service_holding_time,
            k_paths=args.k_paths,
            wandb_log=False,
        )

        the_env = gym.make(args.env_name, **env_args)
        results = []
        for ep in range(args.num_episodes):

            obs = the_env.reset()
            result = nsc_ksp_fdl(the_env)
            results.append(result)

        df = pd.DataFrame(results)
        # Getting the mean reward and mean standard deviation of reward per episode
        df = pd.DataFrame([df.mean().to_dict()])
        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
