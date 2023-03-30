from env.envs.VoneEnv import (
    VoneEnv,
    VoneEnvNodes,
    VoneEnvPaths,
)
from gymnasium.envs.registration import register

register(
    id="vone_Env-v0",
    entry_point="env.envs:VoneEnv",
    order_enforce=False,
)

register(
    id="vone_Env_Nodes-v0",
    entry_point="env.envs:VoneEnvNodes",
    order_enforce=False,
)

register(
    id="vone_Env_Paths-v0",
    entry_point="env.envs:VoneEnvPaths",
    order_enforce=False,
)

