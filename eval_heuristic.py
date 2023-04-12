import gymnasium as gym
from heuristics import *
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import env.envs
import yaml


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
        "--env_file", type=str, help="Environment config file"
    )
    parser.add_argument(
        "--node_heuristic", type=str, help="Node heuristic"
    )
    parser.add_argument(
        "--path_heuristic", type=str, help="Path heuristic"
    )
    args = parser.parse_args()

    conf = yaml.safe_load(Path(args.env_file).read_text())

    data_file = Path(args.output_file)

    data_file.parent.mkdir(exist_ok=True)

    for load in range(args.min_load, args.max_load+1):

        env_args = conf["env_args"]
        env_args["load"] = load
        env_args["wandb_log"] = False

        the_env = gym.make(conf["env_name"], **env_args)
        results = []
        for ep in range(args.num_episodes):

            obs = the_env.reset()
            result, timestep_info_df = run_heuristic(the_env, node_heuristic=args.node_heuristic, path_heuristic=args.path_heuristic)
            results.append(result)

        df = pd.DataFrame(results)
        # Getting the mean reward and mean standard deviation of reward per episode
        df = pd.DataFrame([df.mean().to_dict()])
        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
