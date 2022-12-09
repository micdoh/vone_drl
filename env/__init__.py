from env.envs.VoneEnv import VoneEnv
from gym.envs.registration import register

register(
    id='vone_Env-v0',
    entry_point='env.envs:VoneEnv',
)

register(
    id='vone_Env_Nodes-v0',
    entry_point='env.envs:VoneEnvNodeSelectionOnly',
)

register(
    id='vone_Env_Routing-v0',
    entry_point='env.envs:VoneEnvRoutingOnly',
)