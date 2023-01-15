import gym
import heapq
import random
import networkx as nx
import numpy as np
import pandas as pd
import copy
import logging
import wandb
import stable_baselines3.common
from math import comb, floor
from itertools import combinations, product, islice
from service import Service
from pathlib import Path
from collections import defaultdict
from sympy.utilities.iterables import multiset_permutations
from networktoolbox.NetworkToolkit.Topology import Topology
from heuristics import select_nodes_nsc, select_path_fdl, select_path_ff
from util_funcs import get_nth_item


logger = logging.getLogger(__name__)

# TODO - Functionalise KMC-FF, KMF-FF
# TODO - Plan experiments:
#  1. all-in decision-making (train and run this to get max performance) = 1 option
#  2. nodes FDL/FF with or without masking = 4 options
#  3. routing path-and-slot selection, path-FF, path-FDL = 3 options
#  4. Random policy (could have any of combo of decision stages random) = 1-8 options
#  So, need to

# TODO - Implement scaling of capacity

# TODO - IDEA: Change the reward on the basis of this network utilisation metric.
#  Shaped reward for empty network and simple reward for full network (to allow more honest exploration)
# TODO - Log different failure modes i.e. slot clash, insufficient node capacity, insufficient slots
# TODO - Define Selector class that would allow kSP-FF, -FDL, random, NSC, KMC, etc. to be easily interchanged


fail_messages = {
    "node_mapping": {"code": 1, "message": "Node mapping failure"},
    "path_mapping": {"code": 2, "message": "Path mapping failure"},
    "slot_mapping": {"code": 3, "message": "Slot mapping failure"},
    "node_capacity": {"code": 4, "message": "Insufficient node capacity"},
    "slot_reuse": {"code": 5, "message": "Slot reused in request"},
    "slot_occupied": {"code": 6, "message": "Selected initial slot is occupied"},
    "end_of_band": {"code": 7, "message": "Insufficient neighbouring slots until end of band"},
    "block_size": {"code": 8, "message": "Selected initial slot is of insufficient block size"},
}

class VoneEnvSortedSeparate(gym.Env):
    def __init__(
        self,
        episode_length: int,
        load: int,
        mean_service_holding_time: int,
        k_paths: int = 2,
        topology_path: str = None,
        topology_name: str = "nsfnet",
        num_slots: int = 16,
        node_capacity: int = 5,
        seed: int = 1,
        min_node_cap_request: int = 1,
        max_node_cap_request: int = 2,
        min_slot_request: int = 2,
        max_slot_request: int = 4,
        min_vnet_size: int = 3,
        max_vnet_size: int = 3,
        vnet_size_dist: str = "fixed",
        wandb_log: bool = False,
        routing_choose_k_paths: bool = False,
        ksp_fdl: bool = False,
        ksp_ff: bool = True,
        sort_nodes: bool = True,
    ):
        self.current_time = 0
        self.allocated_Service = []
        self.episode_length = episode_length
        self.services_processed = 0
        self.services_accepted = 0
        self.accepted = False
        self.num_slots = num_slots
        self.node_capacity = node_capacity
        self.num_path_accepted = 0
        self.min_node_cap_request = min_node_cap_request
        self.max_node_cap_request = max_node_cap_request
        self.node_accepted = False
        self.min_slot_request = min_slot_request
        self.max_slot_request = max_slot_request
        self.min_vnet_size = min_vnet_size
        self.max_vnet_size = max_vnet_size
        self.vnet_size_dist = self.vnet_size_distribution(vnet_size_dist)
        self.node_selection_dict = {}
        self.path_selection_dict = {}
        self.link_selection_dict = {}
        self.slot_selection_dict = {}
        self.vnet_cap_request_dict = {}
        self.vnet_bw_request_dict = {}
        self.total_reward = 0
        self.num_resets = 0
        self.wandb_log = wandb_log
        self.current_observation = {}
        self.results = {}
        self.routing_choose_k_paths = routing_choose_k_paths
        self.ksp_fdl = ksp_fdl
        self.ksp_ff = ksp_ff
        self.sort_nodes = sort_nodes
        self.num_selectable_slots = self.num_slots - self.min_slot_request + 1

        self.load = load
        self.mean_service_holding_time = mean_service_holding_time
        self.mean_service_inter_arrival_time = 0
        self.set_load(load, mean_service_holding_time)

        self.k_paths = k_paths

        self.rng = random.Random(seed)

        self.topology_path = None if not topology_path else Path(topology_path)
        self.topology_name = topology_name

        # create topology of substrate network
        self.topology = Topology()
        self.init_topology()
        self.reset_slots()
        self.reset_nodes()
        self.num_nodes = self.topology.topology_graph.number_of_nodes()
        self.num_links = self.topology.topology_graph.number_of_edges()

        # observation_space
        # TODO - Would it be beneficial to also include the original node capacity in the observation?
        #  Recent GNN-DRL VONE paper does so
        self.define_observation_space()

        self.define_action_space()

        # create initialized virtual network observation
        self.current_VN_capacity = np.zeros(self.max_vnet_size, dtype=int)
        self.current_VN_bandwidth = np.zeros(self.max_vnet_size, dtype=int)

        self.info = {
            "acceptance_ratio": None,
            "topology_name": self.topology_name,
            "load": self.load,
        }

    def define_observation_space(self):
        node_resource_capacity = [self.node_capacity + 1] * self.num_nodes
        self.obs_request = gym.spaces.MultiDiscrete(
            (
                (self.max_node_cap_request - self.min_node_cap_request + 1)
                ** self.max_vnet_size,
                (self.max_slot_request - self.min_slot_request + 1)
                ** self.max_vnet_size,
            )
        )
        self.obs_node_capacities = gym.spaces.MultiDiscrete((node_resource_capacity))
        self.obs_slots = gym.spaces.Box(
            low=0, high=1, shape=(self.num_links, self.num_slots), dtype=int
        )
        self.observation_space = gym.spaces.Dict(
            {
                "request": self.obs_request,
                "node_capacities": self.obs_node_capacities,
                "slots": self.obs_slots,
            }
        )

    def define_action_space(self):
        self.generate_link_selection_table()  # Used to map node selection and k-path selections to link selections
        self.generate_vnet_cap_request_tables()
        self.generate_vnet_bw_request_tables()

        # action space sizes are maximum corresponding table size for maximum request size
        # TODO - IDEA: Action space to become node-pair k-path choice (i.e. concurrent action space), such that it can be masked.
        #  This leads to comb(comb(self.num_nodes, 2), self.max_vnet_size) * k total actions for node-path selection
        self.action_space = gym.spaces.MultiDiscrete(
            (
                comb(self.num_nodes, self.max_vnet_size),
                self.k_paths**self.max_vnet_size,
                self.num_selectable_slots ** self.max_vnet_size,
            )
        )

    def reset(self):
        """Called at beginning of each episode"""
        results = {
            "episode_number": self.num_resets,
            "acceptance_ratio": self.services_accepted / self.services_processed
            if self.services_processed > 0
            else 0,
            "mean_reward": self.total_reward / self.services_processed
            if self.services_processed > 0
            else 0,
        }
        logger.warning(f"End of episode {self.num_resets}. Results: {results}")
        if self.wandb_log:
            wandb.log(results)
        self.current_time = 0
        self.allocated_Service = []
        self.services_processed = 0
        self.services_accepted = 0
        self.accepted = False
        self.reset_nodes()
        self.reset_slots()
        self.request_generator()
        self.total_reward = 0
        self.num_resets += 1
        self.results = results
        observation = self.observation()

        return observation

    def init_topology(self):
        """Initialise topology based on contents of config dir"""
        if self.topology_path:
            self.topology_path = self.topology_path / self.topology_name
            self.topology.load_topology(f"{self.topology_path.absolute()}.adjlist")
        elif self.topology_name == "nsfnet":
            self.topology.init_nsf()
        elif self.topology_name == "btcore":
            self.topology.init_btcore()
        elif self.topology_name == "google_b4":
            self.topology.init_google_b4()
        elif self.topology_name == "uknet":
            self.topology.init_uk_net()
        elif self.topology_name == "dtag":
            self.topology.init_dtag()
        elif self.topology_name == "eurocore":
            self.topology.init_EURO_core()
        else:
            raise Exception(
                f"Invalid topology name without specified path: {self.topology_name} \n"
                f"Check config file is correct."
            )
        # Ensure nodes are numbered. Move names to node attribute 'name'.
        self.topology.topology_graph = nx.convert_node_labels_to_integers(
            self.topology.topology_graph, label_attribute="name"
        )

    def reset_slots(self):
        """Set available slots on each link back to default"""
        edge_attrs = {
            edge: {"slots": np.ones(self.num_slots, dtype=int)}
            for edge in self.topology.topology_graph.edges()
        }
        nx.set_edge_attributes(self.topology.topology_graph, values=edge_attrs)

    def reset_nodes(self):
        """Set available capacity on each node back to default"""
        for node in self.topology.topology_graph.nodes:
            self.topology.topology_graph.nodes[node]["capacity"] = self.node_capacity

    def generate_node_selection(self, vnet_size):
        """Populate node_selection_dict with vnet_size: array pairs.
        Array elements indicate node selections, indexed by action space action number"""
        # node selection is sequence e.g. [1, 13, 7] that indicates which nodes will comprise virtual network
        # use combinations as node ordering does not matter
        # dict keyed by vnet size as different node selection table for each vnet size
        return combinations([x for x in range(self.num_nodes)], vnet_size)

    def mask_nodes(self, vnet_size, node_capacities):
        """Return the mask of permitted node actions.
        Check each virtual node requirement in turn,
        find capable nodes,
        get combination of suitable nodes that fit."""
        overall_cap_check = []

        # Check if node can meet v-node requests
        for req_cap in self.current_VN_capacity:
            cap_check = []
            for i, cap in enumerate(node_capacities):
                cap_check.append((0 if cap - req_cap < 0 else 1))
            overall_cap_check.append(cap_check)

        df_cap_check = pd.concat(
            [pd.Series(x) for x in overall_cap_check], axis="columns"
        )
        # count of how many v-nodes a node can satisfy
        df_check_count = df_cap_check.sum(axis="columns")
        # get substrate node combinations and their count of how many v-nodes they can satisfy
        df_check_combinations = pd.DataFrame(
            list(combinations(df_check_count, vnet_size))
        )
        # multiply the total v-nodes that can be satisfied by substrate node combo
        df_check_product = df_check_combinations.prod(axis=1)
        # If product of checks greater than threshold, action is valid
        threshold = vnet_size ** (vnet_size - 1)
        valid_actions = df_check_product.map(lambda x: 1 if x >= threshold else 0)
        return valid_actions

    def mask_paths(self, vnet_size):
        path_table = list(product(range(self.k_paths), repeat=vnet_size))
        return True

    def generate_path_selection(self, vnet_size):
        """Populate path_selection_dict with vnet_size: array pairs.
        Array elements indicate which kth path is taken between each virtual node.
        """
        # k-path selection is sequence of e.g. [0, 1, 2, 1] that indicates which kth path to take between nodes
        # Use Cartesian product of k-path selection because order matters
        # dict keyed by vnet size as different path selection table for each vnet size
        return product(range(self.k_paths), repeat=vnet_size)

    def generate_slot_selection(self, vnet_size):
        """Populate slot_selection_dict with vnet_size: array pairs.
        Array rows are initial slot selection choices"""
        return product(
            range(self.num_selectable_slots), repeat=vnet_size
        )

    def generate_link_selection_table(self):
        """Populate link_selection_dict with node-pair-id: array pairs.
        Array rows give k-shortest path for each node pair"""
        for node_pair in combinations(self.topology.topology_graph.nodes, 2):
            k_paths = self.get_k_shortest_paths(
                self.topology.topology_graph, node_pair[0], node_pair[1], self.k_paths
            )
            self.link_selection_dict[node_pair] = k_paths
            self.link_selection_dict[(node_pair[1], node_pair[0])] = k_paths

    def generate_vnet_cap_request_tables(self):
        for vnet_size in range(self.min_vnet_size, self.max_vnet_size + 1):
            self.vnet_cap_request_dict[vnet_size] = np.array(
                list(
                    product(
                        range(self.min_node_cap_request, self.max_node_cap_request + 1),
                        repeat=vnet_size,
                    )
                )
            )

    def generate_vnet_bw_request_tables(self):
        for vnet_size in range(self.min_vnet_size, self.max_vnet_size + 1):
            self.vnet_bw_request_dict[vnet_size] = np.array(
                list(
                    product(
                        range(self.min_slot_request, self.max_slot_request + 1),
                        repeat=vnet_size,
                    )
                )
            )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action"""
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = get_nth_item(
            self.generate_node_selection(request_size), action[0]
        )
        logger.debug(f" Nodes selected: {nodes_selected}")

        # k_paths selected from the action
        k_path_selected = get_nth_item(
            self.generate_path_selection(request_size), action[1]
        )
        logger.debug(f" Paths selected: {k_path_selected}")

        # initial slot selected from the action
        initial_slot_selected = get_nth_item(
            self.generate_slot_selection(request_size), action[2]
        )
        logger.debug(f" Initial slots selected: {initial_slot_selected}")

        # Return empty dict at end for compatibility with other environments
        return nodes_selected, k_path_selected, initial_slot_selected, {}

    def step(self, action):
        """"""
        logger.debug(f" Timestep  : {self.services_processed}")
        logger.debug(f" Capacity  : {self.current_VN_capacity}")
        logger.debug(f" Bandwidth : {self.current_VN_bandwidth}")

        path_list = []
        path_free = True

        request_size = self.current_VN_capacity.size

        (
            nodes_selected,
            k_path_selected,
            initial_slot_selected,
            fail_info,
        ) = self.select_nodes_paths_slots(action)

        # check substrate node capacity
        cap = self.get_node_capacities(nodes_selected=nodes_selected)

        # rearrange request node-ordering until satisfied that nodes can/cannot satisfy request
        if self.sort_nodes:

            requests = list(zip(self.current_VN_capacity, self.current_VN_bandwidth))
            for n, req in enumerate(multiset_permutations(requests)):
                node_req = [item[0] for item in req]
                bw_req = [item[1] for item in req]
                node_free = (cap >= node_req).all()
                if node_free:
                    self.current_VN_capacity = np.array(node_req)
                    self.current_VN_bandwidth = np.array(bw_req)
                    break

        else:
            # Don't sort nodes if using heuristic node selection
            node_free = (cap >= self.current_VN_capacity).all()

        # make sure different path slots are used & check substrate link BW
        self.num_path_accepted = 0

        # If node check fails, skip path check
        if node_free:

            if fail_info:
                logger.info(fail_info.get("message"))
                path_free = False

            else:
                # 1. Check slots are free
                # 2. Check slots aren't reused in same request

                path_list = self.get_links_from_selection(nodes_selected, k_path_selected)

                for i in range(request_size):

                    # Check if slot is free
                    current_path_free, fail_info = self.is_path_free(
                        path_list[i],
                        initial_slot_selected[i],
                        self.current_VN_bandwidth[i],
                    )

                    if current_path_free:
                        self.num_path_accepted += 1

                    path_free = path_free & current_path_free

                if path_free:  # Check for slot reuse in same request

                    path_free = path_free & self.is_slot_not_reused(
                        path_list, initial_slot_selected, self.current_VN_bandwidth
                    )

                    if not path_free:
                        fail_info = fail_messages["slot_reuse"]

                if fail_info:
                    logger.info(fail_info.get("message"))

        else:
            fail_info = fail_messages["node_capacity"]
            logger.info(fail_info["message"])

        # accepted?
        self.accepted = node_free & path_free
        self.node_accepted = node_free

        # Add service to topology if accepted
        if self.accepted:
            # Holding time
            ht = self.rng.expovariate(1 / self.mean_service_holding_time)
            current_service = Service(
                copy.deepcopy(self.current_time),
                ht,
                nodes_selected,
                copy.deepcopy(self.current_VN_capacity),
                path_list,
                copy.deepcopy(self.current_VN_bandwidth),
                initial_slot_selected,
            )
            self.add_to_list(current_service)
            self.map_service(current_service)
            self.services_accepted += 1

        self.set_load(self.load, self.mean_service_holding_time)
        self.traffic_generator()
        self.request_generator()
        reward = self.reward()
        self.total_reward += reward
        logger.info(f" Reward: {reward}")

        self.services_processed += 1

        observation = self.observation()
        done = self.services_processed == self.episode_length
        info = {
            "acceptance_ratio": self.services_accepted / self.services_processed,
            "topology_name": self.topology_name,
            "load": self.load,
            "mean_service_holding_time": self.mean_service_holding_time,
            "reward": reward,
            "code": 0,
            "message": None,
            "nodes_selected": nodes_selected,
            "paths_selected": k_path_selected,
            "links_selected": path_list,
            "slots_selected": initial_slot_selected,
            **fail_info
        }
        self.current_info = info

        return observation, reward, done, info

    def get_links_from_selection(self, nodes_selected, k_path_selected):
        path_list = []
        for j in range(len(nodes_selected) - 1):
            path_list.append(
                self.link_selection_dict[
                    nodes_selected[j], nodes_selected[j + 1]
                ][k_path_selected[j]]
            )
        path_list.append(
            self.link_selection_dict[nodes_selected[0], nodes_selected[-1]][
                k_path_selected[-1]
            ]
        )
        return path_list

    def vnet_size_distribution(self, dist_name):
        """Set the probability distribution function used to generate the request sizes"""
        if dist_name == "fixed":
            return True#lambda: self.min_vnet_size
        elif dist_name == "random":
            return lambda: self.rng.randint(*(self.min_vnet_size, self.max_vnet_size))
        # TODO - Investigate other possible distributions e.g. realistic traffic
        else:
            raise Exception(
                f"Invalid virtual network size distribution selected: {dist_name}"
            )

    def render(self, mode="human"):
        return self.topology.topology_graph, self.num_slots

    def reward(self):
        """Customisable reward function"""
        if self.accepted:
            reward = 10
        else:
            reward = 0

        return reward

    def get_k_shortest_paths(self, g, source, target, k, weight=None):
        """
        Return list of k-shortest paths from source to target
        Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms
        .simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
        """
        return list(
            islice(nx.shortest_simple_paths(g, source, target, weight=weight), k)
        )

    def is_node_free(self, n, capacity_required):
        """Check node n has sufficient capacity to accommodate request"""
        capacity = self.topology.topology_graph.nodes[n]["capacity"]
        return True if capacity > capacity_required else False

    def get_path_slots(self, path):
        """Return array of slots used by path.

        Args:
            path: List of nodes in path

        Returns:
            path_slots: Array of slots that are either free or occupied along all links in path
        """
        path_slots = [self.topology.topology_graph.edges[path[i], path[i + 1]]["slots"] for i in range(len(path)-1)]
        path_slots = np.stack(path_slots, axis=0)
        path_slots = np.min(path_slots, axis=0)
        return path_slots

    def is_path_free(self, path, initial_slot, num_slots):
        """Check path that initial slot is free and start of block of sufficient capacity"""
        initial_slot = int(initial_slot)
        num_slots = int(num_slots)
        if initial_slot + num_slots > self.num_slots:
            fail_info = fail_messages["end_of_band"]
            return False, fail_info

        path_slots = self.get_path_slots(path)

        if path_slots[initial_slot] == 0:
            fail_info = fail_messages["slot_occupied"]
            return False, fail_info

        elif np.sum(path_slots[initial_slot: initial_slot + num_slots]) < num_slots:
            fail_info = fail_messages["block_size"]
            return False, fail_info

        return True, {}

    def is_slot_not_reused(self, paths, initial_slots, num_slots):
        """Check if requested slots clash in the same request
        1. Check if any links used more than once
        2. Get initial slots and requested number of slots for reused links
        3. Check for overlap"""
        node_pairs = defaultdict(list)
        # Get dictionary of link: [indices of paths that use link]
        for n, path in enumerate(paths):
            for i in range(len(path) - 1):
                s = min([path[i], path[i + 1]])
                d = max([path[i], path[i + 1]])
                node_pairs[(s, d)].append(n)

        for link, n_paths in node_pairs.items():

            # Links reused
            if len(n_paths) > 1:

                # Get slots used in same link for each path then check for duplicates
                req_slots = [
                    slot
                    for n in n_paths
                    for slot in range(initial_slots[n], initial_slots[n] + num_slots[n])
                ]

                if len(req_slots) != len(set(req_slots)):
                    return False

        return True

    def add_to_list(self, service: Service):
        heapq.heappush(
            self.allocated_Service,
            (service.arrival_time + service.holding_time, service),
        )

    def map_service(self, service: Service):
        """Update node and slot capacities"""
        # nodes mapping
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.topology_graph.nodes[node_num][
                "capacity"
            ] -= service.nodes_capacity[i]

        # links mapping
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]

                self.topology.topology_graph.edges[s, d]["slots"] -= slots_occupied

    def set_load(self, load, mean_service_holding_time):
        """Load = Hold time / inter-arrival time"""
        self.mean_service_inter_arrival_time = 1 / float(
            load / float(mean_service_holding_time)
        )

    def release_service(self, service: Service):
        # nodes release
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.topology_graph.nodes[node_num][
                "capacity"
            ] += service.nodes_capacity[i]

        # links release
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]
                self.topology.topology_graph.edges[s, d]["slots"] += slots_occupied

    def traffic_generator(self):
        """
        Method from https://github.com/carlosnatalino/optical-rl-gym/blob/
        fc9a82244602d8efab749fe4391c7ddb4b05dfe7/optical_rl_gym/envs/rmsa_env.py#L280
        """
        at = self.current_time + self.rng.expovariate(
            1 / self.mean_service_inter_arrival_time
        )
        self.current_time = at

        while len(self.allocated_Service) > 0:
            (time, service_to_release) = heapq.heappop(self.allocated_Service)
            if time <= self.current_time:
                self.release_service(service_to_release)
            else:
                heapq.heappush(
                    self.allocated_Service,
                    (
                        service_to_release.arrival_time
                        + service_to_release.holding_time,
                        service_to_release,
                    ),
                )
                break

    def request_generator(self):
        """Generate requested node capacity and link bandwidth"""
        for i in range(len(self.current_VN_capacity)):
            self.current_VN_capacity[i] = self.rng.randint(
                *(self.min_node_cap_request, self.max_node_cap_request)
            )

        for i in range(len(self.current_VN_bandwidth)):
            self.current_VN_bandwidth[i] = self.rng.randint(
                *(self.min_slot_request, self.max_slot_request)
            )

    def get_node_capacities(self, nodes_selected: [int] = None):
        if nodes_selected is not None:
            return np.array(
                [
                    self.topology.topology_graph.nodes[i]["capacity"]
                    for i in nodes_selected
                ]
            )
        else:
            return np.array(
                list(
                    nx.get_node_attributes(
                        self.topology.topology_graph, "capacity"
                    ).values()
                )
            )

    def observation(self):
        # Find row in node request table that matches observation
        node_request_table = self.vnet_cap_request_dict[self.current_VN_capacity.size]
        node_act_int = np.where(
            (node_request_table == self.current_VN_capacity).all(axis=1)
        )[0]

        # Find row in slot request table that matches observation
        slot_request_table = self.vnet_bw_request_dict[self.current_VN_bandwidth.size]
        slot_act_int = np.where(
            (slot_request_table == self.current_VN_bandwidth).all(axis=1)
        )[0]

        slots_matrix = self.get_slots_matrix()

        obs_dict = {
            "request": np.array([*node_act_int, *slot_act_int]),
            "node_capacities": self.get_node_capacities(),
            "slots": slots_matrix.reshape(self.obs_slots.shape),
        }
        self.current_observation = obs_dict
        return obs_dict

    def get_slots_matrix(self):
        return np.array(
            [
                slots for slots in nx.get_edge_attributes(self.topology.topology_graph, "slots").values()
            ]
        )

    def print_topology(self):
        SN_C = np.zeros(self.num_nodes, dtype=int)
        SN_slots = np.zeros((self.num_links, self.num_slots), dtype=int)

        for i in range(len(self.topology.topology_graph.nodes)):
            SN_C[i] = self.topology.topology_graph.nodes[i]["capacity"]

        for i in range(self.num_links):
            SN_slots[i, :] = self.topology.topology_graph.edges[
                np.array(self.topology.topology_graph.edges)[i]
            ]["slots"]

        logger.info(f"SN_C: {SN_C}")
        logger.info(f"SN_slots: {SN_slots}")
        logger.info(f"No. of services: {len(self.allocated_Service)}")


class VoneEnvUnsortedSeparate(VoneEnvSortedSeparate):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sort_nodes = False
        self.action_space = gym.spaces.MultiDiscrete(
            (
                len(self.generate_node_selection(3)),
                self.k_paths**self.max_vnet_size,
                self.num_selectable_slots ** self.max_vnet_size,
            )
        )

    def generate_node_selection(self, vnet_size):
        """Populate node_selection_dict with vnet_size: array pairs.
        Array elements indicate node selections, indexed by action space action number"""
        df = np.array(list(product(range(self.num_nodes), repeat=vnet_size)))
        # Get duplicate node row indices and delete rows
        a = (df[:, 0] == df[:, 1]) | (df[:, 1] == df[:, 2]) | (df[:, 0] == df[:, 2])
        return np.delete(df, np.where(a), axis=0)

    def mask_nodes(self, vnet_size, node_capacities):
        """Return the mask of permitted node actions."""

        node_capacities = dict(enumerate(node_capacities))

        node_selection_table = self.generate_node_selection(self.current_VN_capacity.size)

        node_cap_table = np.vectorize(
            node_capacities.get
        )(
            node_selection_table
        )

        # Set elements to True if node capacity is sufficient
        node_mask = np.greater_equal(node_cap_table, self.current_VN_capacity)

        # Set row to True if all elements True
        node_mask = np.all(node_mask, axis=1)

        return pd.Series(node_mask)


class VoneEnvNodesSorted(VoneEnvSortedSeparate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_action_space(self):
        self.generate_link_selection_table()  # Used to map node selection and k-path selections to link selections
        self.generate_vnet_cap_request_tables()
        self.generate_vnet_bw_request_tables()

        # action space sizes are maximum corresponding table size for maximum request size
        self.action_space = gym.spaces.Discrete(
            comb(self.num_nodes, self.max_vnet_size),
        )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action.
        KSP-FDL or FF"""
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = get_nth_item(
            self.generate_node_selection(request_size), action
        )
        logger.debug(f" Nodes selected: {nodes_selected}")

        (
            k_path_selected,
            initial_slot_selected,
            fail_info
        ) = select_path_fdl(
            self,
            self.topology.topology_graph,
            self.current_VN_bandwidth,
            nodes_selected,
        ) if self.ksp_fdl else select_path_ff(
            self, nodes_selected
        )

        logger.debug(f" Paths selected: {k_path_selected}")
        logger.debug(f" Initial slots selected: {initial_slot_selected}")

        return nodes_selected, k_path_selected, initial_slot_selected, fail_info

    def action_masks(self):
        request_size = self.current_VN_capacity.size
        node_capacities = self.current_observation["node_capacities"]
        node_mask = self.mask_nodes(request_size, node_capacities)
        self.node_mask = node_mask
        return node_mask


class VoneEnvNodesUnsorted(VoneEnvUnsortedSeparate):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sort_nodes = False
        # action space sizes are maximum corresponding table size for maximum request size
        self.action_space = gym.spaces.Discrete(
           len(self.generate_node_selection(3))
        )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action.
        KSP-FDL or FF"""
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = get_nth_item(
            self.generate_node_selection(request_size), action
        )
        logger.debug(f" Nodes selected: {nodes_selected}")

        (
            k_path_selected,
            initial_slot_selected,
            fail_info
        ) = select_path_fdl(
            self,
            self.topology.topology_graph,
            self.current_VN_bandwidth,
            nodes_selected,
        ) if self.ksp_fdl else select_path_ff(
            self, nodes_selected
        )

        logger.debug(f" Paths selected: {k_path_selected}")
        logger.debug(f" Initial slots selected: {initial_slot_selected}")

        return nodes_selected, k_path_selected, initial_slot_selected, fail_info

    def action_masks(self):
        request_size = self.current_VN_capacity.size
        node_capacities = self.current_observation["node_capacities"]
        node_mask = self.mask_nodes(request_size, node_capacities)
        self.node_mask = node_mask
        return node_mask


class VoneEnvRoutingSeparate(VoneEnvSortedSeparate):
    """Observation space is augmented to include selected nodes instead of node capacities.
    Nodes are selected by NSC ranking method."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_observation_space(self):
        self.obs_request = gym.spaces.MultiDiscrete(
            (
                (self.max_node_cap_request - self.min_node_cap_request + 1)
                ** self.max_vnet_size,
                (self.max_slot_request - self.min_slot_request + 1)
                ** self.max_vnet_size,
            )
        )
        self.obs_selected_nodes = gym.spaces.MultiDiscrete(
            (
                self.num_nodes,
                self.num_nodes,
                self.num_nodes,
            )
        )
        node_resource_capacity = [self.node_capacity + 1] * self.num_nodes
        self.obs_node_capacities = gym.spaces.MultiDiscrete(node_resource_capacity)
        self.obs_slots = gym.spaces.Box(
            low=0, high=1, shape=(self.num_links, self.num_slots), dtype=int
        )
        self.observation_space = gym.spaces.Dict(
            {
                "request": self.obs_request,
                "node_capacities": self.obs_node_capacities,
                "selected_nodes": self.obs_selected_nodes,
                "slots": self.obs_slots,
            }
        )

    def define_action_space(self):
        self.generate_link_selection_table()  # Used to map node selection and k-path selections to link selections
        self.generate_vnet_cap_request_tables()
        self.generate_vnet_bw_request_tables()

        # action space sizes are maximum corresponding table size for maximum request size
        if self.routing_choose_k_paths:
            self.action_space = gym.spaces.Discrete(
                self.k_paths**self.max_vnet_size,
            )
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                (
                    self.k_paths**self.max_vnet_size,
                    self.num_selectable_slots ** self.max_vnet_size,
                )
            )

    def observation(self):
        # Find row in node request table that matches observation
        node_request_table = self.vnet_cap_request_dict[self.current_VN_capacity.size]
        node_act_int = np.where(
            (node_request_table == self.current_VN_capacity).all(axis=1)
        )[0]
        nodes_selected, _ = select_nodes_nsc(
            self, self.topology.topology_graph
        )

        # Find row in slot request table that matches observation
        slot_request_table = self.vnet_bw_request_dict[self.current_VN_bandwidth.size]
        slot_act_int = np.where(
            (slot_request_table == self.current_VN_bandwidth).all(axis=1)
        )[0]

        slots_matrix = self.get_slots_matrix()

        obs_dict = {
            "request": np.array([*node_act_int, *slot_act_int]),
            "node_capacities": self.get_node_capacities(),
            "selected_nodes": nodes_selected,
            "slots": slots_matrix.reshape(self.obs_slots.shape),
        }
        self.current_observation = obs_dict
        return obs_dict

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action."""
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = self.current_observation["selected_nodes"]

        if self.routing_choose_k_paths:

            k_paths_selected = get_nth_item(
                self.generate_path_selection(request_size), action
            )

            (
                _,
                initial_slots_selected,
                fail_info,
            ) = select_path_fdl(
                self,
                self.topology.topology_graph,
                self.current_VN_bandwidth,
                nodes_selected,
                k_paths_preselected=k_paths_selected,
            ) if self.ksp_fdl else select_path_ff(
                self, nodes_selected
            )

        else:
            fail_info = {}
            k_paths_selected = get_nth_item(
                self.generate_path_selection(request_size), action[0]
            )
            initial_slots_selected = get_nth_item(
                self.generate_slot_selection(request_size), action[1]
            )

        logger.debug(f" Nodes selected: {nodes_selected}")
        logger.debug(f" Paths selected: {k_paths_selected}")
        logger.debug(f" Slots selected: {initial_slots_selected}")

        return nodes_selected, k_paths_selected, initial_slots_selected, fail_info

    def action_masks(self):
        request_size = self.current_VN_capacity.size
        path_mask = self.mask_paths(request_size)
        self.path_mask = path_mask
        return path_mask


class VoneEnvRoutingCombined(VoneEnvRoutingSeparate):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_action_space(self):
        self.generate_link_selection_table()  # Used to map node selection and k-path selections to link selections
        self.generate_vnet_cap_request_tables()
        self.generate_vnet_bw_request_tables()

        action_space_dims = tuple([self.k_paths*self.num_selectable_slots]) * self.max_vnet_size

        # action space sizes are maximum corresponding table size for maximum request size
        self.action_space = gym.spaces.MultiDiscrete(
            action_space_dims
        )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action."""

        # Get node selection (dependent on number of nodes in request)
        nodes_selected = self.current_observation["selected_nodes"]

        k_paths_selected = [floor(dim / self.num_selectable_slots) for dim in action]
        initial_slots_selected = [dim % self.num_selectable_slots for dim in action]

        logger.debug(f" Nodes selected: {nodes_selected}")
        logger.debug(f" Paths selected: {k_paths_selected}")
        logger.debug(f" Slots selected: {initial_slots_selected}")

        return nodes_selected, k_paths_selected, initial_slots_selected, {}

    def mask_paths(self, vnet_size):
        """Return the mask of permitted path actions."""
        masks = []
        nodes_selected = self.current_observation["selected_nodes"]

        # Get all paths between node pairs
        for k in range(self.k_paths):
            mask = []
            k_paths_selected = [k] * vnet_size
            paths = self.get_links_from_selection(nodes_selected, k_paths_selected)

            # Get all slots on each path
            for i, path in enumerate(paths):
                slots = self.get_path_slots(path)[: self.num_slots - self.current_VN_bandwidth[i]]
                slots = np.pad(slots, (0, self.current_VN_bandwidth[i]), "constant", constant_values=0)[: self.num_selectable_slots]
                mask.append(slots)

            masks.append(np.stack(mask, axis=1))

        total_mask_2d = np.concatenate(masks, axis=0)
        df = pd.DataFrame(total_mask_2d)
        total_mask_1d = np.concatenate([df[n] for n in range(vnet_size)], axis=0).astype(int)

        return total_mask_1d
