import numpy as np
import networkx as nx
import gym
import logging
from copy import deepcopy
from itertools import islice
from env.envs.VoneEnv import VoneEnv

# NSC-KSP-FDL
# method from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7718438
# loop num_episodes:
#   reset the_env
#   read the_env
#   loop episode_length:
#       calculate node rank
#       slot the nodes and do node mapping
#       if node mapping successful:
#           loop each link:
#               (in method_2, we process link with larger capacity requirement first)
#               find all possible slot-blocks in k paths
#               calculate FDL_sum for each slot-blocks
#               do link mapping for one link
#       if link mapping successful:
#           give the action to the_env
#

# TODO - Understand if Yitao implemented another heuristic. If so, what was it?
logger = logging.getLogger(__name__)


def get_k_shortest_paths(g, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))


def find_blocks(slots, num_slots):
    blocks = np.zeros(num_slots)
    n = 0
    for m in range(len(slots)):
        if slots[m] == 1:
            blocks[n] += 1
        else:
            n += 1

    return blocks


def calculate_path_FD(g, path, num_slots):
    """Calculate fragmentation degree for path
    Fragmentation degree for a link is number of blocks divided by number of slots.
    For path, it's sum of FD on the links"""
    FD_sum = 0
    for i in range(len(path) - 1):
        slots = g.edges[path[i], path[i + 1]]['slots']
        VST = sum(slots)
        if VST == 0:
            return 1000
        VSB = find_blocks(slots, num_slots)
        VSB = len(np.where(VSB > 0)[0])
        FD = VSB / VST
        FD_sum += FD

    return FD_sum


def rank_substrate_nsc(topology):
    """Rank substrate nodes by node switching capacity (NSC)
    The NSC of a node is (node capacity * node degree * sum of vacant FSUs on ports)"""
    rank_n_s = []
    for i in topology.nodes:

        vacant_neighboring_slots_sum = 0

        for j in range(topology.degree(i)):

            neighbor = list(topology.adj[i])[j]
            vacant_neighboring_slots_sum += sum(topology.edges[i, neighbor]['slots'])

        switching_capability = topology.degree(i) * vacant_neighboring_slots_sum
        node_capacity = topology.nodes[i]['capacity']
        node_switching_capacity = node_capacity * switching_capability
        rank_n_s.append((node_switching_capacity, i, node_capacity))

    rank_n_s.sort(reverse=True)

    return rank_n_s


def rank_vnet_node_requirements(observation, env):
    """Rank substrate nodes by node switching capacity (NSC)"""
    node_request_int = observation['Vcap_Vbw_Scap'][0]
    node_request_table = env.vnet_cap_request_dict[env.current_VN_capacity.size]
    node_request = node_request_table[node_request_int]
    rank_n_v = []
    for n, cap in enumerate(node_request):
        rank_n_v.append((2 * cap, n, cap))
    rank_n_v.sort(reverse=True)

    return rank_n_v


def get_vnet_bandwidth_requirements(observation, env):
    bw_request_int = observation['Vcap_Vbw_Scap'][1]
    bw_request_table = env.vnet_bw_request_dict[env.current_VN_bandwidth.size]
    return bw_request_table[bw_request_int]


def get_gen_index(gen, value):
    """Get index of generator value that matches"""
    for n, item in enumerate(gen):
        if (item == np.array(value)).all():
            return n


def select_nodes_nsc(env, topology, observation):
    """Return best node choice based on node switching capacity"""
    node_mapping_success = False
    request_size = the_env.current_VN_capacity.size

    action_node = np.zeros(request_size, dtype=int)
    # 1. Rank substrate nodes in descending order of node switching capacity (NSC)
    rank_n_s = rank_substrate_nsc(topology)
    # 2. Rank virtual nodes in descending order of required (capacity x port count)
    rank_n_v = rank_vnet_node_requirements(observation, env)

    # Check that substrate nodes can meet virtual node requirements
    successful_nodes = 0
    for n, node_request in enumerate(rank_n_v):
        if rank_n_s[n][2] >= node_request[2]:
            action_node[rank_n_v[n][1]] = rank_n_s[n][1]
            successful_nodes += 1
    if successful_nodes == request_size:
        node_mapping_success = True
        action_node.sort()  # So that nodes appear in same order as in node selection table row

    return action_node, node_mapping_success


# TODO - The heuristic could be improved by checking for overlap of selected paths and choosing next FDL candidate if so
def select_path_fdl(env, topology, vnet_bandwidth, action_node, k_paths_selected=None):
    """Select k-path and slots to use with fragmentation degree loss (FDL) method.
    If k_path_selected is passed, the k-indices will be used to select paths, with FDL deciding slots.
    Otherwise, both paths and slots are selected based on lowest FDL.
    k_path_selected: List[List[3]]"""
    request_size = the_env.current_VN_capacity.size
    link_mapping_success = True
    action_k_path = np.zeros(request_size, dtype=int)
    action_initial_slots = np.zeros(request_size, dtype=int)
    bw_request_ranked = []

    for i in range(request_size - 1):
        bw_request_ranked.append(
            (vnet_bandwidth[i], action_node[i], action_node[i + 1], i)
        )
    bw_request_ranked.append(
        (vnet_bandwidth[i + 1], action_node[0], action_node[i + 1], i + 1)
    )
    bw_request_ranked.sort(reverse=True)

    for i in range(request_size):
        FDL_candidates = []  # To be populated with candidate path-slots
        bw_req = bw_request_ranked[i][0]
        source = bw_request_ranked[i][1]
        destination = bw_request_ranked[i][2]
        action_index = bw_request_ranked[i][3]

        all_paths = get_k_shortest_paths(topology, source, destination, env.k_paths)

        # If paths already chosen (optional), then only let all paths contain those
        all_paths = [all_paths[k_paths_selected[i]]] if k_paths_selected else all_paths

        # look into single path
        for j in range(len(all_paths)):
            j = k_paths_selected[i] if k_paths_selected else j  # j must equal the k-index of the path
            path = all_paths[0] if k_paths_selected else all_paths[j]
            slots_in_path = []
            for k in range(len(path) - 1):
                slots_in_path.append(topology.edges[path[k], path[k + 1]]['slots'])

            available_slots_in_path = slots_in_path[0]
            for k in range(len(slots_in_path)):
                available_slots_in_path = available_slots_in_path & slots_in_path[k]

            blocks_in_path = find_blocks(available_slots_in_path, the_env.num_slots)
            initial_slots = []
            for k in range(len(blocks_in_path)):
                if blocks_in_path[k] == bw_req:
                    initial_slots.append(k + sum(blocks_in_path[0:k]))
                elif blocks_in_path[k] > bw_req:
                    initial_slots.append(k + sum(blocks_in_path[0:k]))
                    initial_slots.append(k + sum(blocks_in_path[0:k]) + blocks_in_path[k] - bw_req)

            # If no possible slots, try next path
            if len(initial_slots) == 0:
                break
            
            # Calculate FD of path before assigning slots
            FD_before = calculate_path_FD(topology, path, the_env.num_slots)
            initial_slots = np.array(initial_slots, dtype=int)

            # Calaculate FD of path after each possible assignment of slots
            for k in range(len(initial_slots)):
                # Make copy of topology each time before assigning slots
                g = deepcopy(topology)
                
                # Assign slots on each link of path
                for l in range(len(path) - 1):
                    g.edges[path[l], path[l + 1]]['slots'][initial_slots[k]:initial_slots[k] + bw_req] -= 1

                # Calculate FD after assigning slots and hence FDL
                FD_after = calculate_path_FD(g, path, the_env.num_slots)
                FDL_candidates.append((FD_after - FD_before, j, initial_slots[k]))

        # If no paths possible, mapping fails
        if len(FDL_candidates) == 0:
            link_mapping_success = False
            break

        # Sort in order of least FDL
        FDL_candidates.sort()
        action_k_path[action_index] = FDL_candidates[0][1]
        action_initial_slots[action_index] = FDL_candidates[0][2]

        # Update slots in topology clone to inform decision for next index of the request
        path_selected = all_paths[0] if k_paths_selected else all_paths[action_k_path[action_index]]
        for j in range(len(path_selected) - 1):
            topology.edges[
                path_selected[j], path_selected[j + 1]
            ][
                'slots'
            ][
                action_initial_slots[action_index]: action_initial_slots[action_index] + bw_req
            ] -= 1

    action_k_path = k_paths_selected if k_paths_selected else action_k_path

    return link_mapping_success, action_k_path, action_initial_slots


def nsc_ksp_fdl(the_env):

    observation = the_env.reset()

    topology_0, _ = the_env.render()
    topology = deepcopy(topology_0)

    step = 0

    while step < the_env.episode_length:
        step += 1
        link_mapping_success = False
        request_size = the_env.current_VN_capacity.size
        action_k_path = np.zeros(request_size, dtype=int)
        action_initial_slots = np.zeros(request_size, dtype=int)

        # Get node mapping by ranking according to Node Switching Capacity
        action_node, node_mapping_success = select_nodes_nsc(the_env, topology, observation)

        # Get the requested bandwidth between nodes
        vnet_bandwidth = get_vnet_bandwidth_requirements(observation, the_env)

        if node_mapping_success is True:
            link_mapping_success, action_k_path, action_initial_slots = \
                select_path_fdl(the_env, topology, vnet_bandwidth, action_node)

        if link_mapping_success is True:
            # Find row in node selection table that matches desired action
            node_selection_gen = the_env.generate_node_selection(request_size)
            action_node_int = get_gen_index(node_selection_gen, action_node)

            # Find row in path selection table that matches desired action
            path_selection_gen = the_env.generate_path_selection(request_size)
            action_k_path_int = get_gen_index(path_selection_gen, action_k_path)

            # Find row in slot selection table that matches desired action
            slot_selection_gen = the_env.generate_slot_selection(request_size)
            action_initial_slots_int = get_gen_index(slot_selection_gen, action_initial_slots)

        else:
            # TODO - Should make index 0 of action a fail? Or just return a False here and check later?
            action_node_int = action_k_path_int = action_initial_slots_int = 0

        observation, reward, done, info = the_env.step(
            [action_node_int, action_k_path_int, action_initial_slots_int])
        topology_0, _ = the_env.render()
        topology = deepcopy(topology_0)

    logger.warning(info)
    return info


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
    the_env = gym.make('vone_Env-v0', **env_args)

    nsc_ksp_fdl(the_env)

    the_env.close()
