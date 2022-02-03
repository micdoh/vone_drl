import numpy as np
import networkx as nx
import env
import gym
from copy import deepcopy
from itertools import islice


# method from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7718438

def get_k_shortest_paths(g, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))


def find_blocks(slots):
    blocks = np.zeros(8)
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


num_episodes = 1
k_paths = 2
episode_length = 5000
env_args = dict(episode_length=episode_length, load=5,
                mean_service_holding_time=10, k_paths=k_paths, topology_num=3)
the_env = gym.make('network_Env-v0', **env_args)

action_space_size = the_env.action_space
print(action_space_size)

for episodes in range(num_episodes):
    observation = the_env.reset()
    print(observation)
    print(observation['Vcap_Vbw_Scap'][2])

    topology_0, num_of_slots = the_env.render()
    topology = deepcopy(topology_0)
    num_of_nodes = len(topology.nodes)

    for i in range(len(topology.nodes)):
        print(topology.degree(i))
        print(list(topology.adj[i]))

    step = 0

    while step < episode_length:
        print('step', step)
        step += 1
        action_node = np.zeros(3, dtype=int)
        action_k_path = np.zeros(3, dtype=int)
        action_initial_slots = np.zeros(3, dtype=int)

        ## step 1:
        rank_n_s = []
        for i in range(num_of_nodes):
            VSN_sum = 0
            for j in range(topology.degree(i)):
                neighbor = list(topology.adj[i])[j]
                VSN_sum = VSN_sum + sum(topology.edges[i, neighbor]['slots'])

            print(VSN_sum)
            # S_n_s = PN * VSN_sum
            S_n_s = topology.degree(i) * VSN_sum
            C_n_s = topology.nodes[i]['capacity']
            # rank_n_s = S_n_s * C_n_s
            rank_n_s.append((C_n_s * S_n_s, i))

        # in this project, we only consider 3-node 3-link virtual network, so PN of any virtual node is 2
        C_n_v_all = observation['Vcap_Vbw_Scap'][0]
        C_n_v_2 = int(C_n_v_all / 2 ** 2)
        C_n_v_1 = int((C_n_v_all - C_n_v_2 * 2 ** 2) / 2 ** 1)
        C_n_v_0 = int((C_n_v_all - C_n_v_2 * 2 ** 2 - C_n_v_1 * 2 ** 1) / 2 ** 0)
        C_n_v = [C_n_v_0 + 1, C_n_v_1 + 1, C_n_v_2 + 1]
        rank_n_v = []
        for i in range(3):
            rank_n_v.append((2 * C_n_v[i], i))

        ## step 2:
        # sort nodes in a decreasing order
        rank_n_s.sort(reverse=True)
        rank_n_v.sort(reverse=True)

        node_mapping_success = False
        if rank_n_s[0][0] > rank_n_v[0][0]:
            action_node[rank_n_v[0][1]] = rank_n_s[0][1]
            if rank_n_s[1][0] > rank_n_v[1][0]:
                action_node[rank_n_v[1][1]] = rank_n_s[1][1]
                if rank_n_s[2][0] > rank_n_v[2][0]:
                    action_node[rank_n_v[2][1]] = rank_n_s[2][1]
                    node_mapping_success = True

        print(rank_n_s)
        print(rank_n_v)
        print(action_node)

        ## step 3, 4, 5:
        BW_v_all = observation['Vcap_Vbw_Scap'][1]
        BW_v_2 = int(BW_v_all / 2 ** 2)
        BW_v_1 = int((BW_v_all - BW_v_2 * 2 ** 2) / 2 ** 1)
        BW_v_0 = int((BW_v_all - BW_v_2 * 2 ** 2 - BW_v_1 * 2 ** 1) / 2 ** 0)
        BW_v = [BW_v_0 + 2, BW_v_1 + 2, BW_v_2 + 2]

        link_mapping_success = False
        if node_mapping_success is True:
            link_mapping_success = True
            BW_ranked_connection = [(BW_v[0], action_node[0], action_node[1], 0),
                                    (BW_v[1], action_node[0], action_node[2], 1),
                                    (BW_v[2], action_node[1], action_node[2], 2)]
            BW_ranked_connection.sort(reverse=True)
            for i in range(3):
                BW = BW_ranked_connection[i][0]
                source = BW_ranked_connection[i][1]
                destination = BW_ranked_connection[i][2]
                action_index = BW_ranked_connection[i][3]
                all_paths = get_k_shortest_paths(topology, source, destination, k_paths)
                print(all_paths)

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
                        link_mapping_success = False
                        break

                    print(initial_slots)
                    print(FD_before)
                    print(path)

                    for k in range(len(initial_slots)):
                        g = deepcopy(topology)
                        for l in range(len(path) - 1):
                            g.edges[path[l], path[l + 1]]['slots'][initial_slots[k]:initial_slots[k] + BW] -= 1

                        FD_after = calculate_path_FD(g, path)
                        FDL_candidates.append((FD_after - FD_before, j, initial_slots[k]))

                if len(FDL_candidates) == 0:
                    break
                FDL_candidates.sort()
                print(FDL_candidates)
                action_k_path[action_index] = FDL_candidates[0][1]
                action_initial_slots[action_index] = FDL_candidates[0][2]
                path_selected = all_paths[action_k_path[action_index]]
                for j in range(len(path_selected) - 1):
                    topology.edges[path_selected[j], path_selected[j + 1]]['slots'] \
                        [action_initial_slots[action_index]:action_initial_slots[action_index] + BW] -= 1

            if link_mapping_success is True:
                action_node_int = action_node[0] + action_node[1] * num_of_nodes \
                                  + action_node[2] * num_of_nodes ** 2
                action_k_path_int = action_k_path[0] + action_k_path[1] * k_paths \
                                    + action_k_path[2] * k_paths ** 2
                action_initial_slots_int = action_initial_slots[0] + action_initial_slots[1] * num_of_slots \
                                           + action_initial_slots[2] * num_of_slots ** 2
            else:
                action_node_int = 0
                action_k_path_int = 0
                action_initial_slots_int = 0

        observation, reward, done, info = the_env.step([action_node_int, action_k_path_int, action_initial_slots_int])
        topology_0, num_of_slots = the_env.render()
        topology = deepcopy(topology_0)
        print('reward', reward)

print(info)
