import numpy as np
import networkx as nx
import env
import gym
from copy import deepcopy
from itertools import islice

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


def get_k_shortest_paths(g, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))


def find_blocks(slots):
    blocks = np.zeros(16)
    n = 0
    for m in range(len(slots)):
        if slots[m] == 1:
            blocks[n] += 1
        else:
            n += 1

    return blocks


def calculate_path_FD(g, path):
    FD_sum = 0
    for i in range(len(path) - 1):
        slots = g.edges[path[i], path[i + 1]]['slots']
        VST = sum(slots)
        if VST == 0:
            return 1000
        VSB = find_blocks(slots)
        VSB = len(np.where(VSB > 0)[0])
        FD = VSB / VST
        FD_sum += FD

    return FD_sum


def rank_substrate_nsc(topology):
    """Rank substrate nodes by node switching capacity (NSC)"""
    rank_n_s = []
    for i in topology.nodes:
        VSN_sum = 0
        for j in range(topology.degree(i)):
            neighbor = list(topology.adj[i])[j]
            VSN_sum = VSN_sum + sum(topology.edges[i, neighbor]['slots'])

        # S_n_s = PN * VSN_sum
        S_n_s = topology.degree(i) * VSN_sum
        C_n_s = topology.nodes[i]['capacity']
        # rank_n_s = S_n_s * C_n_s
        rank_n_s.append((C_n_s * S_n_s, i, C_n_s))
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


if __name__ == "__main__":

    num_episodes = 1
    k_paths = 2
    episode_length = 5000
    env_args = dict(
        episode_length=episode_length,
        load=6,
        mean_service_holding_time=10,
        k_paths=k_paths,
        wandb_log=False,
    )
    the_env = gym.make('vone_Env-v0', **env_args)

    for episodes in range(num_episodes):
        observation = the_env.reset()

        topology_0, _ = the_env.render()
        topology = deepcopy(topology_0)

        step = 0

        while step < episode_length:
            step += 1
            node_mapping_success = link_mapping_success = False
            request_size = the_env.current_VN_capacity.size
            action_node = np.zeros(request_size, dtype=int)
            action_k_path = np.zeros(request_size, dtype=int)
            action_initial_slots = np.zeros(request_size, dtype=int)

            # 1. Rank substrate nodes in descending order of node switching capacity (NSC)
            rank_n_s = rank_substrate_nsc(topology)
            # 2. Rank virtual nodes in descending order of required (capacity x port count)
            rank_n_v = rank_vnet_node_requirements(observation, the_env)

            # Check that substrate nodes can meet virtual node requirements
            successful_nodes = 0
            for n, node_request in enumerate(rank_n_v):
                if rank_n_s[n][2] >= node_request[2]:
                    action_node[rank_n_v[n][1]] = rank_n_s[n][1]
                    successful_nodes += 1
            if successful_nodes == request_size:
                node_mapping_success = True
                action_node.sort()  # So that nodes appear in same order as in node selection table row

            # step 3, 4, 5:
            vnet_bandwidth = get_vnet_bandwidth_requirements(observation, the_env)

            if node_mapping_success is True:
                link_mapping_success = True
                BW_ranked_connection = []
                for i in range(request_size-1):
                    BW_ranked_connection.append(
                        (vnet_bandwidth[i], action_node[i], action_node[i+1], i)
                    )
                BW_ranked_connection.append(
                    (vnet_bandwidth[i+1], action_node[0], action_node[i+1], i+1)
                )
                BW_ranked_connection.sort(reverse=True)

                for i in range(request_size):
                    BW = BW_ranked_connection[i][0]
                    source = BW_ranked_connection[i][1]
                    destination = BW_ranked_connection[i][2]
                    action_index = BW_ranked_connection[i][3]
                    all_paths = get_k_shortest_paths(topology, source, destination, k_paths)

                    FDL_candidates = []

                    # look into single path
                    for j in range(len(all_paths)):
                        path = all_paths[j]
                        slots_in_path = []
                        for k in range(len(path) - 1):
                            slots_in_path.append(topology.edges[path[k], path[k + 1]]['slots'])

                        available_slots_in_path = slots_in_path[0]
                        for k in range(len(slots_in_path)):
                            available_slots_in_path = available_slots_in_path & slots_in_path[k]

                        blocks_in_path = find_blocks(available_slots_in_path)
                        initial_slots = []
                        for k in range(len(blocks_in_path)):
                            if blocks_in_path[k] == BW:
                                initial_slots.append(k + sum(blocks_in_path[0:k]))
                            elif blocks_in_path[k] > BW:
                                initial_slots.append(k + sum(blocks_in_path[0:k]))
                                initial_slots.append(k + sum(blocks_in_path[0:k]) + blocks_in_path[k] - BW)

                        FD_before = calculate_path_FD(topology, path)
                        initial_slots = np.array(initial_slots, dtype=int)
                        if len(initial_slots) == 0:
                            break

                        for k in range(len(initial_slots)):
                            g = deepcopy(topology)
                            for l in range(len(path) - 1):
                                g.edges[path[l], path[l + 1]]['slots'][initial_slots[k]:initial_slots[k] + BW] -= 1

                            FD_after = calculate_path_FD(g, path)
                            FDL_candidates.append((FD_after - FD_before, j, initial_slots[k]))

                    if len(FDL_candidates) == 0:
                        link_mapping_success = False
                        break

                    FDL_candidates.sort()
                    action_k_path[action_index] = FDL_candidates[0][1]
                    action_initial_slots[action_index] = FDL_candidates[0][2]
                    path_selected = all_paths[action_k_path[action_index]]
                    for j in range(len(path_selected) - 1):
                        topology.edges[
                            path_selected[j], path_selected[j + 1]
                        ][
                            'slots'
                        ][
                        action_initial_slots[action_index]: action_initial_slots[action_index] + BW
                        ] -= 1

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
                action_node_int = action_k_path_int = action_initial_slots_int = 0

            observation, reward, done, info = the_env.step([action_node_int, action_k_path_int, action_initial_slots_int])
            topology_0, _ = the_env.render()
            topology = deepcopy(topology_0)

    print(info)
    the_env.close()
