from env.envs.VoneEnv import (
    VoneEnvUnsortedSeparate,
    VoneEnvSortedSeparate,
    VoneEnvNodesSorted,
    VoneEnvNodesUnsorted,
    VoneEnvRoutingSeparate,
    VoneEnvRoutingCombined,
)
from gym.envs.registration import register

register(
    id="vone_Env_Sorted_Separate-v0",
    entry_point="env.envs:VoneEnvSortedSeparate",
)

register(
    id="vone_Env_Unsorted_Separate-v0",
    entry_point="env.envs:VoneEnvUnsortedSeparate",
)

register(
    id="vone_Env_Nodes_Sorted-v0",
    entry_point="env.envs:VoneEnvNodesSorted",
)

register(
    id="vone_Env_Nodes_Unsorted-v0",
    entry_point="env.envs:VoneEnvNodesUnsorted",
)

register(
    id="vone_Env_Routing_Separate-v0",
    entry_point="env.envs:VoneEnvRoutingSeparate",
)

register(
    id="vone_Env_Routing_Combined-v0",
    entry_point="env.envs:VoneEnvRoutingCombined",
)
