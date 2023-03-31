"""
NSC-KSP-FDL
method from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7718438
loop num_episodes:
  reset the_env
  read the_env
  loop episode_length:
      calculate node rank
      slot the nodes and do node mapping
      if node mapping successful:
          loop each link:
              (in method_2, we process link with larger capacity requirement first)
              find all possible slot-blocks in k paths
              calculate FDL_sum for each slot-blocks
              do link mapping for one link
      if link mapping successful:
          give the action to the_env
"""

import numpy as np
import pandas as pd
import gymnasium as gym
import random
import logging
from copy import deepcopy
from networkx import Graph
from typing import Generator
from util_funcs import get_gen_index


logger = logging.getLogger(__name__)

fail_messages = {
    "node_mapping": {"code": 1, "message": "Node mapping failure"},
    "path_mapping": {"code": 2, "message": "Path mapping failure"},
    "slot_mapping": {"code": 3, "message": "Slot mapping failure"}
}


def find_blocks(slots: np.ndarray) -> np.ndarray:
    """Find starting indices of blocks able to accommodate required slots.

    Args:
        slots: Array of occupied/unoccupied slots on link.

    Returns:
        Binary array with 1 indicating starting slots of blocks able to fit num_slots.
    """
    blocks = np.zeros(len(slots))
    n = 0
    for m in range(len(slots)):
        if slots[m] == 1:  # 1 indicates slot is free
            blocks[n] += 1
        else:
            n += 1  # Increment initial block

    return blocks


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
            return 1000
        block_indices = find_blocks(slots)
        n_blocks = len(np.where(block_indices > 0)[0])
        frag_degree = n_blocks / n_slots
        frag_sum += frag_degree

    return frag_sum


def rank_substrate_nsc(topology: Graph) -> [(int, int, int)]:
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


def rank_vnet_node_requirements(node_request) -> [(int, int, int)]:
    """Rank virtual nodes by node switching capacity (NSC)

    # TODO - N.B. This function only works for 3-node ring virtual topology

    Args:
        node_request: Request virtual node resource capacities.

    Returns:
        Sorted list of nodes and metrics, in descending order of NSC
    """
    rank_n_v = []
    for n, cap in enumerate(node_request):
        rank_n_v.append((2 * cap, n, cap))
    rank_n_v.sort(reverse=True)
    return rank_n_v


def select_nodes_nsc(env: gym.Env, topology: Graph) -> ([int], dict):
    """Return best node choice based on node switching capacity.

    Args:
        env: Gym environment.
        topology: Substrate network graph.

    Returns:
        selected_nodes: List of node indices that have been selected
        node_mapping_success: Bool to indicate successful mapping
    """
    fail_info = fail_messages["node_mapping"]
    request_size = env.current_VN_capacity.size

    selected_nodes = np.zeros(request_size, dtype=int)
    # 1. Rank substrate nodes in descending order of node switching capacity (NSC)
    rank_n_s = rank_substrate_nsc(topology)
    # 2. Rank virtual nodes in descending order of required (capacity x port count)
    rank_n_v = rank_vnet_node_requirements(env.current_VN_capacity)

    # Check that substrate nodes can meet virtual node requirements
    successful_nodes = 0
    for n, node_request in enumerate(rank_n_v):
        substrate_capacity = rank_n_s[n][2]
        requested_capacity = node_request[2]
        if substrate_capacity >= requested_capacity:
            selected_nodes[rank_n_v[n][1]] = rank_n_s[n][1]
            successful_nodes += 1

    # Check for success and clean-up the action
    if successful_nodes == request_size:
        fail_info = {}

    # Sometimes node without sufficient capacity is assigned (even tho one exists)
    # because NSC is calculated based on link capacity * node capacity.
    # So, to make sure there are no duplicate zeroes, we add this step.
    elif np.unique(selected_nodes).size != request_size:
        # Check unique nodes assigned
        selected_nodes = np.array([n for n in range(request_size)])

    return selected_nodes, fail_info


def select_path_fdl(
    env: gym.Env,
    topology_0: Graph,
    vnet_bandwidth: [int],
    selected_nodes: [int],
    k_paths_preselected: [[int]] = None,
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
    fail_info = {}
    topology = deepcopy(topology_0)
    request_size = env.current_VN_capacity.size
    k_paths_selected = np.zeros(request_size, dtype=int)
    initial_slots_selected = np.zeros(request_size, dtype=int)
    bw_request_ranked = []

    # Rank request in descending order of bandwidth
    for i in range(request_size - 1):
        bw_request_ranked.append(
            (vnet_bandwidth[i], selected_nodes[i], selected_nodes[i + 1], i)
        )
    bw_request_ranked.append(
        (vnet_bandwidth[i + 1], selected_nodes[0], selected_nodes[i + 1], i + 1)
    )
    bw_request_ranked.sort(reverse=True)

    # Loop though each part of request
    for i in range(request_size):

        frag_degree_loss_rank = []  # To be populated with candidate path-slots
        bw_req, source, destination, action_index = bw_request_ranked[i]

        all_paths = env.get_k_shortest_paths(topology, source, destination, env.k_paths)

        # If paths already chosen (optional), then only let all paths contain those
        all_paths = [all_paths[k_paths_preselected[i]]] if k_paths_preselected else all_paths

        # look into single path
        for j in range(len(all_paths)):

            j = k_paths_preselected[i] if k_paths_preselected else j  # j must equal the k-index of the path
            path = all_paths[0] if k_paths_preselected else all_paths[j]

            slots_in_path = []
            for k in range(len(path) - 1):
                slots_in_path.append(topology.edges[path[k], path[k + 1]]["slots"])

            available_slots_in_path = slots_in_path[0]
            for k in range(len(slots_in_path)):
                available_slots_in_path = available_slots_in_path & slots_in_path[k]

            blocks_in_path = find_blocks(available_slots_in_path)
            initial_slots = []
            for k in range(len(blocks_in_path)):
                if blocks_in_path[k] == bw_req:
                    # Add initial slot of block
                    initial_slots.append(k + sum(blocks_in_path[0:k]))
                elif blocks_in_path[k] > bw_req:
                    # Add initial slot of block AND final slot of block minus request size
                    initial_slots.append(k + sum(blocks_in_path[0:k]))
                    initial_slots.append(
                        k + sum(blocks_in_path[0:k]) + blocks_in_path[k] - bw_req
                    )

            # If no possible slots, try next path
            if len(initial_slots) == 0:
                break

            # Calculate fragmentation degree of path before assigning slots
            frag_before = calculate_path_frag(topology, path)
            initial_slots = np.array(initial_slots, dtype=int)

            # Calculate frag. degree of path after each possible assignment of slots
            for k in range(len(initial_slots)):
                # Make copy of topology each time before assigning slots
                g = deepcopy(topology)

                # Assign slots on each link of path
                for l in range(len(path) - 1):
                    g.edges[path[l], path[l + 1]]["slots"][
                        initial_slots[k]: initial_slots[k] + bw_req
                    ] -= 1

                # Calculate frag. degree after assigning slots and hence FDL
                frag_after = calculate_path_frag(g, path)
                frag_degree_loss_rank.append(
                    (frag_after - frag_before, j, initial_slots[k])
                )

            # Only need to check one path if preselected, so break
            if k_paths_preselected:
                break

        # If no paths possible, mapping fails
        if len(frag_degree_loss_rank) == 0:
            fail_info = fail_messages["path_mapping"]
            break

        # Sort in order of least FDL
        frag_degree_loss_rank.sort()
        k_paths_selected[action_index] = frag_degree_loss_rank[0][1]
        initial_slots_selected[action_index] = frag_degree_loss_rank[0][2]

        # Update slots in topology clone to inform decision for next index of the request
        path_selected = (
            all_paths[0] if k_paths_preselected else all_paths[k_paths_selected[action_index]]
        )
        for j in range(len(path_selected) - 1):
            topology.edges[path_selected[j], path_selected[j + 1]]["slots"][
                initial_slots_selected[action_index]: initial_slots_selected[action_index]
                + bw_req
            ] -= 1

    k_paths_selected = k_paths_preselected if k_paths_preselected else k_paths_selected

    return k_paths_selected, initial_slots_selected, fail_info


def select_path_ff(env: gym.Env, nodes_selected: [int]):
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
    request_size = env.current_VN_capacity.size
    # Loop through requests
    for i_req in range(request_size):

        current_path_free = False

        # Check each k-path
        for k in range(env.k_paths):

            if current_path_free:
                break

            # Get Links
            path_list = env.get_links_from_selection(nodes_selected, [k]*request_size)

            for i_slot in range(env.num_slots - env.current_VN_bandwidth[i_req]):
                # Check each slot in turn to see if free
                current_path_free, fail_info = env.is_path_free(
                    path_list[i_req], i_slot, env.current_VN_bandwidth[i_req]
                )

                if current_path_free:
                    k_paths_selected.append(k)
                    initial_slots_selected.append(i_slot)
                    break

        if not current_path_free:
            k_paths_selected.append(0)
            initial_slots_selected.append(0)

    return k_paths_selected, initial_slots_selected, fail_info


def nsc_ksp_fdl(the_env: gym.Env, combined: bool = True):
    """

    Args:
        the_env: Gym environment.
        combined: whether to use combined path-slot selection or separate
        i.e. combined dimension for path selection or separate dimension per path

    Returns:
        results: dict of load, reward, std
    """
    observation = the_env.reset()

    topology_0, _ = the_env.render()
    topology = deepcopy(topology_0)
    info = {}
    info_list = []

    step = 0

    while step < the_env.episode_length:
        step += 1
        request_size = the_env.current_VN_capacity.size

        # Get node mapping by ranking according to Node Switching Capacity
        action_node, fail_info = select_nodes_nsc(the_env, topology)

        # Get the requested bandwidth between nodes
        vnet_bandwidth = the_env.current_VN_bandwidth

        if not fail_info:

            action_k_path, action_initial_slots, fail_info = select_path_fdl(
                the_env, topology, vnet_bandwidth, action_node
            )

            # The environment requires actions to be presented in the same
            # integer format as the agent uses.
            # Store those integers here
            action_ints = []

            # Find row in node selection table that matches desired action
            node_selection_gen = the_env.generate_node_selection(request_size)
            action_ints.append(get_gen_index(node_selection_gen, action_node))

            # If using combined path-slot selection
            if combined:

                for i in range(request_size):
                    action_ints.append(action_k_path[i]*the_env.num_selectable_slots + action_initial_slots[i])

            else:
                # Find row in path selection table that matches desired action
                path_selection_gen = the_env.generate_path_selection(request_size)
                action_ints.append(get_gen_index(path_selection_gen, action_k_path))

                # Find row in slot selection table that matches desired action
                slot_selection_gen = the_env.generate_slot_selection(request_size)
                action_ints.append(get_gen_index(slot_selection_gen, action_initial_slots))

        else:

            # TODO - Should make index 0 of action a fail? Or just return a False here and check later?
            action_ints = [0] + [0] * request_size if combined else [0, 0, 0]

        observation, reward, done, _, info = the_env.step(
            action_ints
        )
        topology_0, _ = the_env.render()
        topology = deepcopy(topology_0)
        info_list.append(info)

    logger.info(info)

    df = pd.DataFrame(info_list)

    results = {"load": the_env.load, "reward": df["reward"].mean(), "std": df["reward"].std()}
    logger.info(results)

    return results


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
