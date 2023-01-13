import gym
from env.envs.VoneEnv import VoneEnvSortedSeparate, VoneEnvUnsortedSeparate
from heuristics import nsc_ksp_fdl
import pandas as pd
import os
from pathlib import Path

if __name__ == "__main__":

    data_file = Path("/Users/michaeldoherty/git/vone_drl/runs/nsc_ksp_fdl/nsc_ksp_fdl_load_9.csv")

    data_file.parent.mkdir(exist_ok=True)

    num_episodes = 20
    k_paths = 5
    episode_length = 5000
    env_args = dict(
        episode_length=episode_length,
        load=9,
        mean_service_holding_time=10,
        k_paths=k_paths,
        wandb_log=False,
        sort_nodes=False,
    )

    the_env = gym.make("vone_Env_Unsorted_Separate-v0", **env_args)

    for ep in range(num_episodes):

        obs = the_env.reset()
        info_list = nsc_ksp_fdl(the_env, sort=False)

        df = pd.DataFrame(info_list)
        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
