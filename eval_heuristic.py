import gym
from env.envs.VoneEnv import VoneEnv
from heuristics import nsc_ksp_fdl

if __name__ == "__main__":

    num_episodes = 1
    k_paths = 5
    episode_length = 5000
    env_args = dict(
        episode_length=episode_length,
        load=6,
        mean_service_holding_time=10,
        k_paths=k_paths,
        wandb_log=False,
    )

    the_env = gym.make("vone_Env-v0", **env_args)

    nsc_ksp_fdl(the_env)

    the_env.close()
