from gym.envs.registration import register

register(
    id='network_Env-v0',
    entry_point='env.envs:MyEnv',
)
