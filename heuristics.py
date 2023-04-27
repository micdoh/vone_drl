import math
import networkx as nx
import numpy as np
import pandas as pd
import gymnasium as gym
import logging
from copy import deepcopy
from networkx import Graph
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv



logger = logging.getLogger(__name__)

fail_messages = {
    "node_mapping": {"code": 1, "message": "Node mapping failure"},
    "path_mapping": {"code": 2, "message": "Path mapping failure"},
    "slot_mapping": {"code": 3, "message": "Slot mapping failure"}
}


def find_blocks(slots: np.ndarray, size_until_end: bool = True) -> np.ndarray:
    """
    Find starting indices of blocks of consecutive unoccupied slots in the input array.

    Args:
        slots: A binary numpy array representing the occupied (1) and unoccupied (0) slots on a link.
        size_until_end: (optional) If True, the output array will contain the remaining slots until the
            end of the block of consecutive unoccupied slots, starting at the corresponding index in the
            input array. If False, the output array will contain the length of the block of consecutive
            unoccupied slots, starting at the corresponding index in the input array.

    Returns:
        A numpy array of the same length as the input array, where each element indicates the remaining
        slots until the end of the block of consecutive unoccupied slots, starting at the corresponding
        index in the input array. If the element is 0, it means that there is no block of unoccupied slots
        starting at that index.
    """
    padded_array = np.concatenate(([0], slots, [0]))
    change_indices = np.where(padded_array[:-1] != padded_array[1:])[0]

    if len(change_indices) % 2 != 0:
        change_indices = np.append(change_indices, len(slots))

    lengths = change_indices[1::2] - change_indices[::2]

    output_array = np.zeros_like(slots)
    for start, length in zip(change_indices[::2], lengths):
        output_array[start:start + length] = np.arange(length, 0, -1) if size_until_end else length
    return output_array


def count_blocks(slots: np.ndarray) -> int:
    """
    Count the number of blocks of consecutive unoccupied slots in the input array.

    Args:
        slots: A binary numpy array representing the occupied (1) and unoccupied (0) slots on a link.

    Returns:
        An integer representing the number of blocks of consecutive unoccupied slots in the input array.
    """
    padded_array = np.concatenate(([0], slots, [0]))
    change_indices = np.where(padded_array[:-1] != padded_array[1:])[0]

    if len(change_indices) % 2 != 0:
        change_indices = np.append(change_indices, len(slots))

    num_blocks = len(change_indices) // 2

    return num_blocks


def select_random_action(env: gym.Env, action_index: int = None) -> int or np.ndarray:
    """Randomly sample the action space.

    Args:
        env: The environment.
        action_index: (optional) The index of the multidiscrete action space value to return.

    Return:
        Action array or indexed value
    """
    action = env.action_space.sample()
    if action_index:
        return action[action_index]
    return action


def find_consecutive_ones(arr):
    if not np.any(arr):  # If the input array contains no ones
        return np.array([0])

    # Add a zero at the beginning and the end to easily detect consecutive sequences
    padded_arr = np.concatenate(([0], arr, [0]))

    # Find the indices of ones and compute differences between consecutive indices
    ones_indices = np.where(padded_arr == 1)[0]
    diff_indices = np.concatenate((np.diff(ones_indices), np.array([0])))

    # Identify the end of consecutive sequences by finding where the difference is not 1
    end_of_sequences = np.where(diff_indices != 1)[0]

    # Calculate the lengths of consecutive sequences
    lengths = np.diff(np.concatenate(([0], end_of_sequences + 1)))

    return lengths


def calculate_path_frag(topology: Graph, path: [int]) -> int:
    """Calculate fragmentation degree for path.

    Fragmentation degree for a link is number of blocks divided by number of slots.
    For a path, sum the FD of its constituent links.

    Args:
        topology: Substrate network graph.
        path: List of nodes comprising the path.

    Returns:
        Sum of the fragmentation degree for the path,
        or 1000 (large number) if any link fully occupied.
    """
    frag_sum = 0
    for i in range(len(path) - 1):
        slots = topology.edges[path[i], path[i + 1]]["slots"]
        n_slots = sum(slots)
        if n_slots == 0:
            return 1e6  # Large number
        n_blocks = count_blocks(slots)
        frag_degree = n_blocks / n_slots
        frag_sum += frag_degree

    return frag_sum


def rank_nodes_nsc(topology: Graph) -> [(int, int, int)]:
    """Rank substrate nodes by node switching capacity (NSC).

    The NSC of a node is (node capacity * node degree * sum of vacant FSUs on ports)

    Args:
        topology: Substrate network graph.

    Returns:
        List of tuples of (NSC, node index, node capacity), ranked by NSC
    """
    rank_nsc = []
    for i in topology.nodes:

        vacant_neighboring_slots_sum = 0

        # Get sum of neighbouring vacant slots
        for j in range(topology.degree(i)):

            neighbor = list(topology.adj[i])[j]
            vacant_neighboring_slots_sum += sum(topology.edges[i, neighbor]["slots"])

        # Calculate NSC
        switching_capability = topology.degree(i) * vacant_neighboring_slots_sum
        node_capacity = topology.nodes[i]["capacity"]
        node_switching_capacity = node_capacity * switching_capability
        rank_nsc.append((node_switching_capacity, i, node_capacity))

    # Rank nodes in descending order of NSC
    rank_nsc.sort(reverse=True)

    return rank_nsc


def rank_v_nodes_nsc(node_request, adjacency_list=((0, 1), (1, 2), (2, 0))) -> [(int, int, int)]:
    """Rank virtual nodes by node switching capacity (NSC)

    Args:
        node_request: Request virtual node resource capacities.
        adjacency_list: List of tuples defining virtual network topology.

    Returns:
        List of tuples of (NSC, node index, node capacity), ranked by NSC
    """
    rank = {}
    for n, cap in enumerate(node_request):
        rank[n] = [0, cap]
        for adj in adjacency_list:
            if n in adj:
                rank[n][0] += cap
    rank = [(nsc[0], node, nsc[1]) for node, nsc in rank.items()]
    rank.sort(reverse=True)
    return rank


def select_path_fdl(
        env: gym.Env,
        selected_nodes: [int],
        adjacency_list: tuple = ((0, 1), (1, 2), (2, 0))
) -> ([int], [int], bool):
    """Select k-path and slots to use with fragmentation degree loss (FDL) method.

    If k_paths_preselected is passed, the k-indices will be used to select paths, with FDL deciding slots.
    Otherwise, both paths and slots are selected based on lowest FDL.

    Args:
        env: Gym environment.
        topology_0: Substrate network graph.
        vnet_bandwidth: Requested bandwidth.
        selected_nodes:

    Returns:
        k_paths_selected: List of elected k-path indices.
        initial_slots_selected: List of selected initial slots.
        fail_info: Dict containing comment on failure mode.
    """
    # TODO - Update this to use an adjacency list to get request size
    fail_info = {}
    topology = deepcopy(env.topology.topology_graph)
    vnet_bandwidth = env.current_VN_bandwidth
    request_size = len(adjacency_list)
    k_paths_selected = np.zeros(request_size, dtype=int)
    initial_slots_selected = np.zeros(request_size, dtype=int)

    # Rank request in descending order of bandwidth
    bw_request_ranked = sorted(
        [(vnet_bandwidth[i], selected_nodes[i], selected_nodes[i + 1], i) for i in range(request_size - 1)] +
        [(vnet_bandwidth[-1], selected_nodes[0], selected_nodes[-1], request_size - 1)],
        reverse=True
    )

    # Loop though each part of request
    for i in range(request_size):

        frag_degree_loss_rank = []  # To be populated with candidate path-slots
        bw_req, source, destination, action_index = bw_request_ranked[i]

        all_paths = env.link_selection_dict[(source, destination)]

        # look into single path
        for k in range(len(all_paths)):

            path = all_paths[k]

            path_slots = env.get_path_slots(path, topology)

            blocks_in_path = find_blocks(path_slots)

            initial_slots = []
            m = 0
            while m < (len(blocks_in_path)):
                block_size = blocks_in_path[m]
                if block_size >= bw_req:
                    # Add initial slot of block
                    initial_slots.append(m)
                    if block_size > bw_req:
                        # Add final slot of block minus request size
                        initial_slots.append(m + block_size - bw_req)
                    m += block_size
                else:
                    m += 1

            # If no possible slots, try next path
            if len(initial_slots) == 0:
                break

            # Calculate fragmentation degree of path before assigning slots
            frag_before = calculate_path_frag(topology, path)
            initial_slots = np.array(initial_slots, dtype=int)

            # Calculate frag. degree of path after each possible assignment of slots
            for j in range(len(initial_slots)):
                # Make copy of topology each time before assigning slots
                g = deepcopy(topology)

                # Assign slots on each link of path
                for l in range(len(path) - 1):
                    g.edges[path[l], path[l + 1]]["slots"][
                        initial_slots[j]: initial_slots[j] + bw_req
                    ] -= 1

                # Calculate frag. degree after assigning slots and hence FDL
                frag_after = calculate_path_frag(g, path)
                frag_degree_loss_rank.append(
                    (frag_after - frag_before, k, initial_slots[j])
                )

        # If no paths possible, mapping fails
        if len(frag_degree_loss_rank) == 0:
            fail_info = fail_messages["path_mapping"]
            break

        # Sort in order of least FDL
        frag_degree_loss_rank.sort()
        k_paths_selected[action_index] = frag_degree_loss_rank[0][1]
        initial_slots_selected[action_index] = frag_degree_loss_rank[0][2]

        if i == request_size - 1:
            return k_paths_selected, initial_slots_selected, fail_info
        else:
            # Update slots in topology clone to inform decision for next index of the request
            path_selected = all_paths[k_paths_selected[action_index]]
            for j in range(len(path_selected) - 1):
                slots = topology.edges[path_selected[j], path_selected[j + 1]]["slots"]
                topology.edges[path_selected[j], path_selected[j + 1]]["slots"][
                    initial_slots_selected[action_index]: initial_slots_selected[action_index] + bw_req
                ] -= 1

    return k_paths_selected, initial_slots_selected, fail_info


def select_path_ff(env: gym.Env, nodes_selected: [int], adjacency_list = ((0, 1), (1, 2), (2, 0))):
    """kSP-FF

    Args:
        env: Gym environment.
        nodes_selected: Indices of nodes that comprise virtual network.

    Returns:
        k_path_selected: List of k-path indices.
        initial_slot_selected: List of starting slot indices for each path.
        fail_info: Dict containing comment on failure mode.
    """
    # 1. For each node pair, get k-path, iterate through slots until fits, if no fit then go to next k-path
    fail_info = {}
    k_paths_selected = []
    initial_slots_selected = []
    topology = deepcopy(env.topology.topology_graph)
    # Retrieve all k_paths for the selected nodes
    path_dict = {k: env.get_links_from_selection(nodes_selected, [k]*len(adjacency_list), adjacency_list=adjacency_list) for k in range(env.k_paths)}
    # Loop through requests
    for i_req, adj in enumerate(adjacency_list):

        bw = env.current_VN_bandwidth[i_req]

        current_path_free = False

        # Check each k-path
        for k in range(env.k_paths):

            path = path_dict[k][i_req]

            if current_path_free:
                break

            for i_slot in range(env.num_slots - bw):
                # Check each slot in turn to see if free
                current_path_free, fail_info = env.is_path_free(topology, path, i_slot, bw)

                if current_path_free:
                    k_paths_selected.append(k)
                    initial_slots_selected.append(i_slot)
                    # Assign slots on each link of path on topology copy
                    for l in range(len(path) - 1):
                        topology.edges[path[l], path[l + 1]]["slots"][
                        i_slot: i_slot + bw
                        ] -= 1
                    break

        if not current_path_free:
            k_paths_selected = [0] * len(adjacency_list)
            initial_slots_selected = [0] * len(adjacency_list)
            break

    return k_paths_selected, initial_slots_selected, fail_info


def select_path_msp_ef(env: gym.Env, nodes_selected: [int], adjacency_list = ((0, 1), (1, 2), (2, 0))):
    """Modified shortest path exact-fit.

    Args:
        env: Gym environment.
        nodes_selected: Indices of nodes that comprise virtual network.

    Returns:
        k_path_selected: List of k-path indices.
        initial_slot_selected: List of starting slot indices for each path.
        fail_info: Dict containing comment on failure mode.
    """
    k_paths_selected = []
    initial_slots_selected = []
    topology = deepcopy(env.topology.topology_graph)
    # Retrieve all k_paths for the selected nodes
    path_dict = {k: env.get_links_from_selection(nodes_selected, [k]*len(adjacency_list), adjacency_list=adjacency_list) for k in range(env.k_paths)}
    # Loop through requests
    for i_req, adj in enumerate(adjacency_list):

        bw = env.current_VN_bandwidth[i_req]

        tightest_path_fit, fail_info = calc_shortest_weighted_path(topology, path_dict, i_req, bw)
        #tightest_path_fit, fail_info = calc_tightest_mf_path_weight(env, topology, path_dict, i_req, bw)

        k_paths_selected.append(tightest_path_fit)

        path = path_dict[tightest_path_fit][i_req]

        tightest_slot_fit, fail_info = find_tightest_slot(env, topology, path, bw)

        initial_slots_selected.append(tightest_slot_fit)

        if fail_info:
            k_paths_selected = [0] * len(adjacency_list)
            initial_slots_selected = [0] * len(adjacency_list)
            break

        print(f"k_path: {tightest_path_fit}")
        print(f"slot: {tightest_slot_fit}")
        print(f"bw: {bw}")
        print(f"path: {path}")

        # Assign slots on each link of path on topology copy
        for l in range(len(path) - 1):
            slots = topology.edges[path[l], path[l + 1]]["slots"]
            print(f"Slots: {slots}")
            topology.edges[path[l], path[l + 1]]["slots"][
            tightest_slot_fit: tightest_slot_fit + bw
            ] -= 1
            print(f"Slots after: {topology.edges[path[l], path[l + 1]]['slots']}")

    return k_paths_selected, initial_slots_selected, fail_info


def calc_tightest_mf_path_weight(env, graph, path_dict, i_req, bw):
    # Check each k-path
    weights = []
    for k in range(env.k_paths):
        path = path_dict[k][i_req]
        path_slots = env.get_path_slots(path, graph)
        min_block_size = min(find_consecutive_ones(path_slots))
        weight = max(((min_block_size - bw) / max(min_block_size, 1)) + 1e-5, 0)
        weight = 1e6 if weight == 0 else weight
        weights.append(weight*(len(path)-1))
    tightest_fit_index = weights.index(min(weights))
    print(f"Path {k}: {path}, weights: {weights}, tightest fit: {tightest_fit_index}")
    fail_info = fail_messages["path_mapping"] if all(x == 0 for x in weights) else {}
    return tightest_fit_index, fail_info


def calc_shortest_weighted_path(graph, path_dict, i_req, bw):
    min_weight = float('inf')
    shortest_path = None
    for k, paths in path_dict.items():
        path = paths[i_req]
        weights = []
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        for edge in edges:
            slots = graph.edges[edge[0], edge[1]]["slots"]
            min_block_size = min(find_consecutive_ones(slots))
            weight = max(((min_block_size - bw) / max(min_block_size, 1)) + 1e-5, 0)
            weight = 1e6 if weight == 0 else weight
            weights.append(weight)
        path_weight = sum(weights)
        if path_weight < min_weight:
            min_weight = path_weight
            shortest_path = k
            fail_info = fail_messages["path_mapping"] if all(x == 0 for x in weights) else {}
        print(f"Path {k}: {path}, weights: {weights}, path_weight: {path_weight}, min_weight: {min_weight}")

    return shortest_path, fail_info

def find_nth_block_index(slots, n):
    occurrences = np.where(slots >= 1)[0]
    return occurrences[n - 1] if len(occurrences) >= n else -1


def find_tightest_slot(env, graph, path, bw):
    path_slots = env.get_path_slots(path, graph)
    print(f"path_slots: {path_slots}")
    block_sizes = find_blocks(path_slots, size_until_end=False)
    weights = np.maximum(((block_sizes - bw) / np.maximum(block_sizes, 1)) + 1e-5, 0)
    weights = np.where(weights == 0, 1e6, weights)
    tightest_fit_index = np.argmin(weights)
    print(f"block_sizes: {block_sizes}, weights: {weights}, tightest_fit_index: {tightest_fit_index}")
    fail_info = fail_messages["slot_mapping"] if all(x == 0 for x in weights) else {}
    return int(tightest_fit_index), fail_info


def rank_v_nodes_lrc(node_request, bw_request, adjacency_list=((0, 1), (1, 2), (2, 0))):
    """Calculate the Local Resource Capacity of a node.

    Args:
        graph: NetworkX graph.
        node: Node to calculate LRC of.
        bw_requests: Bandwidth requests of all virtual links.

    Returns:
        lrc: Local Resource Capacity of node.
    """
    rank = {}
    for n, cap in enumerate(node_request):
        rank[n] = [0, cap]
        for bw, adj in zip(bw_request, adjacency_list):
            if n in adj:
                rank[n][0] += cap*bw
    rank = [(lrc[0], node, lrc[1]) for node, lrc in rank.items()]
    rank.sort(reverse=True)
    return rank


def rank_nodes_calrc(graph, bw_request):
    """Calculate the Consecutiveness-Aware Local Resource Capacity of a node.

    Args:
        graph: NetworkX graph.
        node: Node to calculate LRC of.
        bw_requests: Bandwidth requests of all virtual links.

    Returns:
        ranking: List of tuples of (node_index, calrc) i.e. ranking of substrate nodes
        based on Consecutiveness-Aware Local Resource Capacity.
    """
    rank = []
    bw_request = set(bw_request)
    for node in graph.nodes:
        calrc = 0
        capacity = graph.nodes[node]["capacity"]
        for _, _, edge_data in graph.edges(node, data=True):
            MSBC_sizes = find_consecutive_ones(edge_data["slots"])
            for i in bw_request:
                calrc += np.sum(MSBC_sizes - i + 1)
        rank.append((capacity * calrc, node, capacity))
    rank.sort(reverse=True)
    return rank


def calc_link_asc(link):
    """Calculate the continuity degree of available spectrum of a link.

    Args:
        link: Link to calculate ASC of.

    Returns:
        asc: Continuity degree of available spectrum of a link.
    """
    slots = link["slots"]
    available_slot_sum = np.sum(slots)
    num_slots = len(slots)
    num_blocks = count_blocks(slots)
    return available_slot_sum / (num_slots * num_blocks)


def rank_nodes_mf(graph, betweenness):
    """Rank nodes based on matching factor.

    Args:
        graph: NetworkX graph.
        betweenness: Betweenness centrality of nodes.

    Returns:
        ranking: List of tuples of (mf, node_index, capacity) i.e. ranking of substrate nodes
    """
    rank = []
    for node in graph.nodes:
        capacity = graph.nodes[node]["capacity"]
        cap_factor = (capacity / sum(graph.nodes[n]["capacity"] for n in graph.nodes))
        asc_factor = sum([calc_link_asc(link) for _, _, link in graph.edges(node, data=True)]) / \
                     sum([calc_link_asc(link) for _, _, link in graph.edges(data=True)])
        mrcc = cap_factor * asc_factor
        mf = mrcc * math.exp(-betweenness[node]+1)
        rank.append((mf, node, capacity))
    rank.sort(reverse=True)
    print(f"Ranking snodes: {rank}")
    return rank


def rank_v_nodes_mrr(cap_request, bw_request, adjacency_list=((0, 1), (1, 2), (2, 0))):
    """Calculate the multidimensional resources requirement of virtual nodes and rank them.

    Args:
        cap_request: List of capacity requests of virtual nodes.
        bw_request: List of bandwidth requests of virtual links.
        adjacency_list: List of tuples of adjacent virtual nodes.

    Returns:
        ranking: List of tuples of (mrr, node_index, capacity) i.e. ranking of virtual nodes
    """
    rank = {}
    for n, cap in enumerate(cap_request):
        rank[n] = [0, cap]
        adj_bw_requests = sum([bw_request[j] for j, adj in enumerate(adjacency_list) if n in adj])
        rank[n][0] = (cap/np.sum(cap_request)) * (adj_bw_requests / np.sum(bw_request))
    rank = [(mrr[0], node, mrr[1]) for node, mrr in rank.items()]
    rank.sort(reverse=True)
    print(f"Ranking vnodes: {rank}")
    return rank


def select_nodes(env: gym.Env, topology: Graph, heuristic: str = "nsc", betweenness: dict = {}) -> ([int], dict):
    """Select nodes for mapping based on a heuristic.

    Args:
        env: Gym environment.
        topology: Substrate network graph.
        heuristic: Heuristic to use for node selection.
        betweenness: Betweenness centrality of nodes (optional)

    Returns:
        selected_nodes: List of node indices that have been selected
        node_mapping_success: Bool to indicate successful mapping
    """
    fail_info = fail_messages["node_mapping"]
    request_size = env.current_VN_capacity.size

    selected_nodes = np.zeros(request_size, dtype=int)

    if heuristic == "nsc":
        rank_n_s = rank_nodes_nsc(topology)
        rank_n_v = rank_v_nodes_nsc(env.current_VN_capacity)
    elif heuristic == "calrc":
        rank_n_s = rank_nodes_calrc(topology, env.current_VN_bandwidth)
        rank_n_v = rank_v_nodes_lrc(env.current_VN_capacity, env.current_VN_bandwidth)
    elif heuristic == "tmr":
        rank_n_s = rank_nodes_mf(topology, betweenness)
        rank_n_v = rank_v_nodes_mrr(env.current_VN_capacity, env.current_VN_bandwidth)
    else:
        raise ValueError(f"Heuristic '{heuristic}' not supported.")

    # Check that substrate nodes can meet virtual node requirements
    successful_nodes = 0
    for v_node in rank_n_v:
        for i, s_node in enumerate(rank_n_s):
            substrate_capacity = s_node[2]
            requested_capacity = v_node[2]
            if substrate_capacity >= requested_capacity:
                selected_nodes[v_node[1]] = s_node[1]
                successful_nodes += 1
                rank_n_s.pop(i)  # Remove selected node from list
                break

    # Check for success and clean-up the action
    if successful_nodes == request_size:
        fail_info = {}

    # Sometimes no substrate node assigned to a virtual node.
    # So, to make sure there are no duplicate zeroes, we add this step.
    elif np.unique(selected_nodes).size != request_size:
        # Check unique nodes assigned
        selected_nodes = np.array([n for n in range(request_size)])

    return selected_nodes, fail_info

def run_heuristic(the_env: gym.Env, node_heuristic: str = "nsc", path_heuristic: str = "ff", combined: bool = True):
    """

    Args:
        the_env: Gym environment.
        combined: whether to use combined path-slot selection or separate
        i.e. combined dimension for path selection or separate dimension per path

    Returns:
        results: dict of load, reward, std
    """
    observation = the_env.reset()
    the_env = the_env.envs[0] if isinstance(the_env, DummyVecEnv or SubprocVecEnv) else the_env
    info = {}
    info_list = []
    step = 0
    betweenness = nx.betweenness_centrality(the_env.topology.topology_graph) if node_heuristic == "tmr" else None

    while step < the_env.episode_length:
        step += 1
        topology = deepcopy(the_env.topology.topology_graph)
        request_size = the_env.current_VN_capacity.size
        node_table = the_env.node_selection_dict[request_size]
        path_table = the_env.path_selection_dict[request_size]
        slot_table = the_env.slot_selection_dict[request_size]

        # Get node selection
        action_node, fail_info = select_nodes(the_env, topology, node_heuristic, betweenness=betweenness)

        if not fail_info:

            if path_heuristic == "ff":
                action_k_path, action_initial_slots, fail_info = select_path_ff(the_env, action_node)
            elif path_heuristic == "fdl":
                action_k_path, action_initial_slots, fail_info = select_path_fdl(the_env, action_node)
            elif path_heuristic == "msp_ef":
                action_k_path, action_initial_slots, fail_info = select_path_msp_ef(the_env, action_node)
            else:
                raise ValueError(f"Path heuristic '{path_heuristic}' not supported.")

            # The environment requires actions to be presented in the same integer format as the agent uses.
            # Store those integers here
            action_ints = []

            # Find row in node selection table that matches desired action
            action_ints.append(np.where(np.all(node_table == action_node, axis=1))[0][0])

            # If using combined path-slot selection
            if combined:
                path_action_ints = [action_k_path[i]*the_env.num_selectable_slots + action_initial_slots[i]
                                    for i in range(request_size)]
                action_ints.extend(path_action_ints)

            else:
                # Find row in path selection table that matches desired action
                action_ints.append(np.where(np.all(path_table == action_k_path, axis=1))[0][0])

                # Find row in slot selection table that matches desired action
                action_ints.append(np.where(np.all(slot_table == action_initial_slots, axis=1))[0][0])

        else:
            # If no path found, select random action
            action_ints = [0] + [0] * request_size if combined else [0, 0, 0]

        observation, reward, done, _, info = the_env.step(
            action_ints
        )
        info_list.append(info)

    logger.info(info)

    df = pd.DataFrame(info_list)

    results = {
        "load": the_env.load,
        "reward": df["reward"].mean(),
        "blocking": 1-df["acceptance_ratio"].iloc[-1],
    }
    logger.info(results)

    return results, df
