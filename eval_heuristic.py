import gym
from heuristics import nsc_ksp_fdl
import pandas as pd
import numpy as np
import os
from pathlib import Path
from env.envs.VoneEnv import (
    VoneEnvUnsortedSeparate,
    VoneEnvSortedSeparate,
    VoneEnvNodesSorted,
    VoneEnvNodesUnsorted,
    VoneEnvRoutingSeparate,
    VoneEnvRoutingCombined,
    VoneEnvUnsortedCombined,
)

if __name__ == "__main__":

    data_file = Path("/eval/nsc_ksp_fdl/nsc_ksp_fdl_df.csv")

    data_file.parent.mkdir(exist_ok=True)

    for load in range(1, 16):

        num_episodes = 3
        k_paths = 5
        episode_length = 5000
        env_args = dict(
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=10,
            k_paths=k_paths,
            wandb_log=False,
            sort_nodes=False,
        )

        the_env = gym.make("vone_Env_Unsorted_Separate-v0", **env_args)
        results = []
        for ep in range(num_episodes):

            obs = the_env.reset()
            result = nsc_ksp_fdl(the_env, sort=False)
            results.append(result)

        df = pd.DataFrame(results)
        df = pd.DataFrame([df.mean().to_dict()])  # Getting the mean reward and mean standard deviation of reward per episode
        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
