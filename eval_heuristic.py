import gymnasium as gym
from heuristics import *
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import env.envs
import yaml
from datetime import datetime
from util_funcs import make_env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", default="", type=str, help="Path to output csv file"
    )
    parser.add_argument(
        "--timestep_output_file", default="", type=str, help="Path to output csv file for timestep info"
    )
    parser.add_argument(
        "--min_load", default=1, type=int, help="Minimum load"
    )
    parser.add_argument(
        "--max_load", default=15, type=int, help="Maximum load"
    )
    parser.add_argument(
        "--load_step", default=1, type=int, help="Increment load by this amount"
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
    parser.add_argument(
        "--random", action="store_true", help="Use random policy for evaluation"
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
    args = parser.parse_args()

    conf = yaml.safe_load(Path(args.env_file).read_text())

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    data_file = Path(args.output_file).parent / start_time / Path(args.output_file).name
    timestep_data_file = data_file.parent / Path(args.timestep_output_file).name
    data_file.parent.mkdir(exist_ok=True)

    for load in range(args.min_load, args.max_load+1, args.load_step):

        env_args = conf["env_args"]
        env_args["load"] = load
        env_args["wandb_log"] = False

        the_env = gym.make(conf["env_name"], **env_args).seed(load) if not args.random \
            else DummyVecEnv([make_env(conf["env_name"], load, **env_args)])
        results = []
        for ep in range(args.num_episodes):

            obs = the_env.reset()
            result, timestep_info_df = run_heuristic(
                the_env,
                node_heuristic=args.node_heuristic,
                path_heuristic=args.path_heuristic
            ) if not args.random else run_random_masked_heuristic(
                the_env,
                action_interpreter=args.action_interpreter,
                masking_steps=args.multistep_masking_n_steps,
                multistep_masking_attr=args.multistep_masking_attr
            )
            results.append(result)
            # Write timestep info to file
            timestep_info_df.to_csv(timestep_data_file, mode='a', header=not os.path.exists(timestep_data_file))

        df = pd.DataFrame(results)

        # Getting the std dev of reward and blocking prob. across episodes
        reward_std = df["reward"].std()
        blocking_std = df["blocking"].std()
        df["reward_std"] = reward_std
        df["blocking_std"] = blocking_std

        # Getting the mean reward and blocking prob. across episodes
        df = pd.DataFrame([df.mean().to_dict()])

        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
