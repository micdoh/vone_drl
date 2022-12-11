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
from math import comb
from functools import reduce
from itertools import combinations, product, islice, permutations
from service import Service
from pathlib import Path
from collections import defaultdict
from sympy.utilities.iterables import multiset_permutations
from networktoolbox.NetworkToolkit.Topology import Topology
from nsc_ksp_fdl import select_nodes_nsc, select_path_fdl


logger = logging.getLogger(__name__)

# TODO - Implement routing-only environment: try both path-selection-only and slot-and-path selection
# TODO - Functionalise KSP-FF, KSP-FDL, KMC-FF, KMF-FF
# TODO - Plan experiments

# TODO - Implement scaling of capacity
# TODO - Implement invalid action masking: 1. find valid rows in tables and corresponding indexes, 2. generate mask
# TODO - Accommodate transparent VONE case (continuity constraint)

# TODO - Log arrival rate and load as metrics
# TODO - Log different failure modes i.e. slot clash, insufficient node capacity, insufficient slots
# TODO - Return the intra-episode acceptance rate, not just post post-episode.
#  i.e. plot acceptance rate against number of request received. Could also incorporate into reward.


def get_nth_item(gen, n):
    """Return the nth item from a generator"""
    return next(islice(gen, n, None), None)


class VoneEnv(gym.Env):

    def __init__(self,
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
                 vnet_size_dist: str = 'fixed',
                 wandb_log: bool = False,
                 routing_choose_k_paths: bool = False,
                 nodes_ksp_fdl: bool = False,
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
        self.nodes_ksp_fdl = nodes_ksp_fdl

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
        node_resource_capacity = [self.node_capacity+1]*self.num_nodes
        # TODO - Would it be beneficial to also include the original node capacity in the observation?
        #  Recent GNN-DRL VONE paper does so
        self.obs_request_and_nrc = gym.spaces.MultiDiscrete(
            (
                (self.max_node_cap_request - self.min_node_cap_request + 1) ** self.max_vnet_size,
                (self.max_slot_request - self.min_slot_request + 1) ** self.max_vnet_size,
                *node_resource_capacity
            )
        )
        self.obs_slots = gym.spaces.Box(low=0, high=1, shape=(self.num_slots, self.num_links), dtype=int)
        self.observation_space = gym.spaces.Dict(
            {'Vcap_Vbw_Scap': self.obs_request_and_nrc,
             'slots': self.obs_slots}
        )

        self.define_action_space()

        # create initialized virtual network observation
        self.current_VN_capacity = np.zeros(self.max_vnet_size, dtype=int)
        self.current_VN_bandwidth = np.zeros(self.max_vnet_size, dtype=int)

    def define_action_space(self):
        self.generate_link_selection_table()  # Used to map node selection and k-path selections to link selections
        self.generate_vnet_cap_request_tables()
        self.generate_vnet_bw_request_tables()

        # action space sizes are maximum corresponding table size for maximum request size
        # TODO - IDEA: Action space to become node-pair k-path choice (i.e. concurrent action space),
        #  such that it can be masked.
        #  This leads to comb(comb(self.num_nodes, 2), self.max_vnet_size) * k total actions for node-path selection
        self.action_space = gym.spaces.MultiDiscrete(
            (
                comb(self.num_nodes, self.max_vnet_size),
                self.k_paths**self.max_vnet_size,
                (self.num_slots-self.min_slot_request+1) ** self.max_vnet_size,
            )
        )

    def reset(self):
        """Called at beginning of each episode"""
        results = {
                "episode_number": self.num_resets,
                "acceptance_ratio": self.services_accepted / self.services_processed if self.services_processed > 0 else 0,
                "mean_reward": self.total_reward / self.services_processed if self.services_processed > 0 else 0,
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
        elif self.topology_name == 'nsfnet':
            self.topology.init_nsf()
        elif self.topology_name == 'btcore':
            self.topology.init_btcore()
        elif self.topology_name == 'google_b4':
            self.topology.init_google_b4()
        elif self.topology_name == 'uknet':
            self.topology.init_uk_net()
        elif self.topology_name == 'dtag':
            self.topology.init_dtag()
        elif self.topology_name == 'eurocore':
            self.topology.init_EURO_core()
        else:
            raise Exception(f'Invalid topology name without specified path: {self.topology_name} \n'
                            f'Check config file is correct.')
        # Ensure nodes are numbered. Move names to node attribute 'name'.
        self.topology.topology_graph = nx.convert_node_labels_to_integers(
            self.topology.topology_graph,
            label_attribute='name'
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
            self.topology.topology_graph.nodes[node]['capacity'] = self.node_capacity

    def generate_node_selection(self, vnet_size):
        """Populate node_selection_dict with vnet_size: array pairs.
        Array elements indicate node selections, indexed by action space action number"""
            # node selection is sequence e.g. [1, 13, 7] that indicates which nodes will comprise virtual network
            # use combinations as node ordering does not matter
            # dict keyed by vnet size as different node selection table for each vnet size
        return combinations([x for x in range(self.num_nodes)], vnet_size)

    def generate_node_mask(self, vnet_size, node_capacities):
        """Return the mask of permitted node actions.
        Check each virtual node requirement in turn,
        find capable nodes,
        get combination of suitable nodes that fit."""
        overall_cap_check = []

        # Check if node can meet v-node requests
        for req_cap in self.current_VN_capacity:
            cap_check = []
            for i, cap in enumerate(node_capacities):
                cap_check.append((0 if cap-req_cap < 0 else 1))
            overall_cap_check.append(cap_check)

        df_cap_check = pd.concat([pd.Series(x) for x in overall_cap_check], axis='columns')
        # count of how many v-nodes a node can satisfy
        df_check_count = df_cap_check.sum(axis='columns')
        # get substrate node combinations and their count of how many v-nodes they can satisfy
        df_check_combinations = pd.DataFrame(list(combinations(df_check_count, vnet_size)))
        # multiply the total v-nodes that can be satisfied by substrate node combo
        df_check_product = df_check_combinations.prod(axis=1)
        # If product of checks greater than threshold, action is valid
        threshold = vnet_size**(vnet_size-1)
        valid_actions = df_check_product.map(lambda x: 1 if x >= threshold else 0)
        return valid_actions

    def generate_path_mask(self, vnet_size):
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
        return product(range(self.num_slots-self.min_slot_request+1), repeat=vnet_size)

    def generate_link_selection_table(self):
        """Populate link_selection_dict with node-pair-id: array pairs.
        Array rows give k-shortest path for each node pair"""
        for node_pair in combinations(self.topology.topology_graph.nodes, 2):
            k_paths = self.get_k_shortest_paths(self.topology.topology_graph, node_pair[0], node_pair[1], self.k_paths)
            self.link_selection_dict[node_pair] = k_paths

    def generate_vnet_cap_request_tables(self):
        for vnet_size in range(self.min_vnet_size, self.max_vnet_size + 1):
            self.vnet_cap_request_dict[vnet_size] = np.array(
                list(
                    product(range(self.min_node_cap_request, self.max_node_cap_request+1), repeat=vnet_size)
                )
            )

    def generate_vnet_bw_request_tables(self):
        for vnet_size in range(self.min_vnet_size, self.max_vnet_size + 1):
            self.vnet_bw_request_dict[vnet_size] = np.array(
                list(
                    product(range(self.min_slot_request, self.max_slot_request+1), repeat=vnet_size)
                )
            )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action"""
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = get_nth_item(self.generate_node_selection(request_size), action[0])
        logger.info(f' Nodes selected: {nodes_selected}')

        # k_paths selected from the action
        k_path_selected = get_nth_item(self.generate_path_selection(request_size), action[1])
        logger.info(f' Paths selected: {k_path_selected}')

        # initial slot selected from the action
        initial_slot_selected = get_nth_item(self.generate_slot_selection(request_size), action[2])
        logger.info(f' Initial slots selected: {initial_slot_selected}')

        # Return empty dict at end for compatibility with other environments
        return nodes_selected, k_path_selected, initial_slot_selected, {}

    def step(self, action):
        """step function has two part.
        In Part1, it checks whether the action meets the constraints,
        In Part2, it pushes the time to the moment just before the new request arriving, and send it to the agent

        #####################################################################################################
        virtual network contains 3 nodes and 3 links in this case.
        The observation and action spaces choose from the number of combinations of nodes, k_path, initial slots,
        the combination could be treated as integer number in different bases.
        For example, 3 virtual nodes (VN0, VN1, VN2) are mapping to 5 substrate nodes (SN0, SN1, SN2, SN3, SN4),
              VN0 might be mapped to any one of 5 substrate nodes, and same for VN1 and VN2,
              thus, the mapping action could be represented by integer within the interval [000,444],
              which is a Hexadecimal (base 5) number.
              The number of integers within the interval is just the action space.
              If we do base conversion, we can transfer Decimal number to Hexadecimal number,
              and hence know the nodes.
        Same for k_path action and initial slot action"""
        logger.info(f' Timestep  : {self.services_processed}')
        logger.info(f' Capacity  : {self.current_VN_capacity}')
        logger.info(f' Bandwidth : {self.current_VN_bandwidth}')

        node_free = path_free = True

        request_size = self.current_VN_capacity.size

        nodes_selected, k_path_selected, initial_slot_selected, fail_info = self.select_nodes_paths_slots(action)

        # check substrate node capacity
        cap = np.array([self.topology.topology_graph.nodes[i]['capacity'] for i in nodes_selected])

        # rearrange request node-ordering until satisfied that nodes can/cannot satisfy request
        requests = list(zip(self.current_VN_capacity[::], self.current_VN_bandwidth))
        for n, req in enumerate(multiset_permutations(requests)):
            node_req = [item[0] for item in req]
            bw_req = [item[1] for item in req]
            node_free = (cap >= node_req).all()
            if node_free:
                self.current_VN_capacity = np.array(node_req)
                self.current_VN_bandwidth = np.array(bw_req)
                break

        # make sure different path slots are used & check substrate link BW
        self.num_path_accepted = 0

        # If node check fails, skip path check
        if node_free:

            if fail_info:
                logger.info(fail_info.get('message'))
                path_free = False

            else:
                # 1. Check slots are free
                # 2. Check slots aren't reused in same request

                for i in range(request_size):

                    path_list = []
                    for j in range(len(nodes_selected) - 1):
                        path_list.append(
                            self.link_selection_dict[
                                nodes_selected[j],
                                nodes_selected[j+1]
                            ]
                            [
                                k_path_selected[j]
                            ]
                        )
                    path_list.append(
                        self.link_selection_dict[nodes_selected[0], nodes_selected[-1]]
                        [k_path_selected[-1]]
                    )

                    current_path_free = self.is_path_free(
                        path_list[i],
                        initial_slot_selected[i],
                        self.current_VN_bandwidth[i]
                    )

                    if current_path_free:
                        self.num_path_accepted += 1

                    path_free = path_free & current_path_free

                path_free = path_free & self.is_slot_not_reused(path_list, initial_slot_selected, self.current_VN_bandwidth)

        else:
            #print(f'\n\n\n\n\n\n {nodes_selected, self.current_VN_capacity, [self.current_observation["Vcap_Vbw_Scap"][2:][x] for x in nodes_selected]}')
            #print(self.current_observation["Vcap_Vbw_Scap"][2:])
            #print(list(self.generate_node_selection(request_size)))
            #print(' \n\n\n\n\n\n')
            logger.info(' Request failure: Insufficient capacity at selected node')

        # accepted?
        self.accepted = node_free & path_free
        self.node_accepted = node_free

        if self.accepted:
            ht = self.rng.expovariate(1 / self.mean_service_holding_time)
            current_service = Service(copy.deepcopy(self.current_time), ht,
                                      nodes_selected, copy.deepcopy(self.current_VN_capacity),
                                      path_list, copy.deepcopy(self.current_VN_bandwidth),
                                      initial_slot_selected)
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
        info = {'P_accepted': self.services_accepted / self.services_processed,
                'topology_name': self.topology_name,
                'load': self.load}

        return observation, reward, done, info

    def vnet_size_distribution(self, dist_name):
        """Set the probability distribution function used to generate the request sizes"""
        if dist_name == 'fixed':
            return lambda: self.min_vnet_size
        elif dist_name == 'random':
            return lambda: self.rng.randint(*(self.min_vnet_size, self.max_vnet_size))
        # TODO - Investigate other possible distributions e.g. realistic traffic
        else:
            raise Exception(f'Invalid virtual network size distribution selected: {dist_name}')

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
        return list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))

    def is_node_free(self, n, capacity_required):
        """Check node n has sufficient capacity to accommodate request"""
        capacity = self.topology.topology_graph.nodes[n]['capacity']
        return True if capacity > capacity_required else False

    def is_path_free(self, path, initial_slot, num_slots, log=True):
        """Check path that initial slot is free and start of block of sufficient capacity"""
        if initial_slot + num_slots > self.num_slots:
            if log:
                logger.info(' Request failure: Selected initial slot does not have '
                            'sufficient neighbouring slots until end of band')
            return False

        path_slots = np.ones(self.num_slots, dtype=int)
        for i in range(len(path) - 1):
            path_slots = path_slots & self.topology.topology_graph.edges[path[i], path[i + 1]]['slots']

        if path_slots[initial_slot] == 0:
            if log:
                logger.info(' Request failure: Selected initial slot is occupied')
            return False

        elif np.sum(path_slots[initial_slot: initial_slot+num_slots]) < num_slots:
            if log:
                logger.info(' Request failure: Selected initial slot is of insufficient block size')
            return False

        return True

    def is_slot_not_reused(self, paths, initial_slots, num_slots):
        """Check if requested slots clash in the same request
        1. Check if any links used more than once
        2. Get initial slots and requested number of slots for reused links
        3. Check for overlap"""
        node_pairs = defaultdict(list)
        # Get dictionary of link: [path_indices]
        for n, path in enumerate(paths):
            for i in range(len(path)-1):
                node_pairs[(path[i], path[i+1])].append(n)

        for link, n_paths in node_pairs.items():

            # Links reused
            if len(n_paths) > 1:

                # Get slots used in same link for each path then check for duplicates
                req_slots = [slot for n in n_paths for slot in range(initial_slots[n], initial_slots[n]+num_slots[n])]
                if len(req_slots) != len(set(req_slots)):
                    logger.info(' Request failure: Assigned slots clash in reused link')
                    return False

        return True

    def add_to_list(self, service: Service):
        heapq.heappush(self.allocated_Service, (service.arrival_time + service.holding_time, service))

    def map_service(self, service: Service):
        # nodes mapping
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.topology_graph.nodes[node_num]['capacity'] -= service.nodes_capacity[i]

        # links mapping
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]

                self.topology.topology_graph.edges[s, d]['slots'] -= slots_occupied

    def set_load(self, load, mean_service_holding_time):
        """Load = Hold time / inter-arrival time"""
        self.mean_service_inter_arrival_time = 1 / float(load / float(mean_service_holding_time))

    def release_service(self, service: Service):
        # nodes release
        for i in range(len(service.nodes)):
            node_num = service.nodes[i]
            self.topology.topology_graph.nodes[node_num]['capacity'] += service.nodes_capacity[i]

        # links release
        for i in range(len(service.path)):
            slots_occupied = np.zeros(self.num_slots, dtype=int)
            for j in range(service.links_BW[i]):
                slots_occupied[service.links_IS[i] + j] = 1

            for k in range(len(service.path[i]) - 1):
                s = service.path[i][k]
                d = service.path[i][k + 1]
                self.topology.topology_graph.edges[s, d]['slots'] += slots_occupied

    def traffic_generator(self):
        """
        Method from https://github.com/carlosnatalino/optical-rl-gym/blob/
        fc9a82244602d8efab749fe4391c7ddb4b05dfe7/optical_rl_gym/envs/rmsa_env.py#L280
        """
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        while len(self.allocated_Service) > 0:
            (time, service_to_release) = heapq.heappop(self.allocated_Service)
            if time <= self.current_time:
                self.release_service(service_to_release)
            else:
                heapq.heappush(self.allocated_Service,
                               (service_to_release.arrival_time + service_to_release.holding_time,
                                service_to_release)
                               )
                break

    def request_generator(self):
        """Generate requested node capacity and link bandwidth"""
        for i in range(len(self.current_VN_capacity)):
            self.current_VN_capacity[i] = self.rng.randint(*(self.min_node_cap_request, self.max_node_cap_request))

        for i in range(len(self.current_VN_bandwidth)):
            self.current_VN_bandwidth[i] = self.rng.randint(*(self.min_slot_request, self.max_slot_request))

    def get_node_capacities(self):
        return np.array(list(nx.get_node_attributes(self.topology.topology_graph, 'capacity').values()))

    def observation(self):
        # VN observation + SN observation
        obs_dict = {'Vcap_Vbw_Scap': np.zeros(len(self.current_VN_capacity), dtype=int),
                    'slots': np.zeros(self.num_slots * self.num_links, dtype=int)}

        # Find row in node request table that matches observation
        node_request_table = self.vnet_cap_request_dict[self.current_VN_capacity.size]
        Vcap_int = np.where((node_request_table == self.current_VN_capacity).all(axis=1))[0]

        # Find row in slot request table that matches observation
        slot_request_table = self.vnet_bw_request_dict[self.current_VN_bandwidth.size]
        Vbw_int = np.where((slot_request_table == self.current_VN_bandwidth).all(axis=1))[0]

        # Get node capacities from graph attribute
        Scap_array = self.get_node_capacities()

        obs_dict['Vcap_Vbw_Scap'] = np.array([*Vcap_int, *Vbw_int, *Scap_array])

        Sslots_matrix = np.array([self.topology.topology_graph.adj[edge[0]][edge[1]]['slots'] for edge in self.topology.topology_graph.edges])
        obs_dict['slots'] = Sslots_matrix.reshape(self.obs_slots.shape)
        self.current_observation = obs_dict
        return obs_dict

    def valid_action_mask(self):
        # Action selection is node selection, path selection, slot selection
        # TODO - make current request size an attribute
        request_size = self.current_VN_capacity.size
        node_capacities = self.current_observation["Vcap_Vbw_Scap"][2:]
        slot_capacities = self.current_observation["slots"]
        # Masks need to be [true, false, ...] lists, with one list for each discrete action space.
        # So need to return [[], [], []] from this function
        # The logic to generate the lists can be derived from the logic used to generate the action tables maybe...
        # Check nodes
        node_table = np.array(list(self.generate_node_selection(request_size)))
        node_mask = self.generate_node_mask(request_size, node_capacities)
        # Check slots: if block out slot selection and, if slots blocked on a path, block
        slot_table = np.array(list(self.generate_slot_selection(request_size)))
        slot_mask = self.generate_slot_mask(request_size, slot_capacities)
        # Notes on masking:
        # The decision-making sequence of nodes-path-slots requires the path selection to be known
        # before the slot selection can be fully masked. Without knowing the selected path (and therefore links),
        # the only way the slot selection can be masked is by masking any initial slots on individual links that
        # don't have sufficient contiguous slots. So, essentially this is the only way masking can be achieved atm.
        # To achieve true masking, we'll need to open up the policy network code to access the sequential
        # decision-making, get the selected path, mask the slots, select an option, and then optionally,
        # if no slots available for the chosen path, mask that path and renormalise the probabilities for the k-1
        # paths remaining and re-mask the slots again until an action is possible.

        path_table = np.array(list(self.generate_path_selection(request_size)))
        link_table = self.link_selection_dict
        #index = np.searchsorted(data, values)
        return [True]

    def check_network_utilisation(self):
        # TODO - IDEA: Change the reward on the basis of this metric.
        #  Shaped reward for empty network and simple reward for full network (to allow more honest exploration)
        "Define metric for how 'full' the substrate network is"
        return True

    def print_topology(self):
        SN_C = np.zeros(self.num_nodes, dtype=int)
        SN_slots = np.zeros((self.num_links, self.num_slots), dtype=int)

        for i in range(len(self.topology.topology_graph.nodes)):
            SN_C[i] = self.topology.topology_graph.nodes[i]['capacity']

        for i in range(self.num_links):
            SN_slots[i, :] = self.topology.topology_graph.edges[
                np.array(self.topology.topology_graph.edges)[i]
            ]['slots']

        logger.info(f'SN_C: {SN_C}')
        logger.info(f'SN_slots: {SN_slots}')
        logger.info(f'No. of services: {len(self.allocated_Service)}')


class VoneEnvNodeSelectionOnly(VoneEnv):

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
        KSP-FF"""
        fail_info = {}
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected = get_nth_item(self.generate_node_selection(request_size), action)
        logger.info(f' Nodes selected: {nodes_selected}')

        if self.nodes_ksp_fdl:
            # kSP-FDL
            link_mapping_success, k_path_selected, initial_slot_selected = \
                select_path_fdl(
                    self,
                    self.topology.topology_graph,
                    self.current_VN_bandwidth,
                    nodes_selected,
                )
            if not link_mapping_success:
                fail_info = {"message": " Request failure: Suitable path not found by kSP-FF"}

        else:
            # kSP-FF
            # 1. For each node pair, get k-path, iterate through slots until fits, if no fit then go to next k-path
            # Could use find_blocks here to speed things up?
            k_path_selected = []
            initial_slot_selected = []
            # Loop through requests
            for i_req in range(request_size):

                current_path_free = False

                # Check each k-path
                for k in range(self.k_paths):

                    if current_path_free:
                        break

                    # Get Links
                    path_list = []
                    for j in range(len(nodes_selected) - 1):
                        path_list.append(
                            self.link_selection_dict[
                                nodes_selected[j],
                                nodes_selected[j + 1]
                            ]
                            [
                                k
                            ]
                        )
                    path_list.append(self.link_selection_dict[nodes_selected[0], nodes_selected[-1]][k])

                    for i_slot in range(self.num_slots-self.current_VN_bandwidth[i_req]):
                        # Check each slot in turn to see if free
                        current_path_free = self.is_path_free(
                            path_list[i_req],
                            i_slot,
                            self.current_VN_bandwidth[i_req],
                            log=False
                        )

                        if current_path_free:
                            k_path_selected.append(k)
                            initial_slot_selected.append(i_slot)
                            break

                if not current_path_free:
                    k_path_selected.append(-1)
                    initial_slot_selected.append(-1)
                    fail_info = {"message": " Request failure: Suitable path not found by kSP-FF"}

        logger.info(f' Paths selected: {k_path_selected}')
        logger.info(f' Initial slots selected: {initial_slot_selected}')

        return nodes_selected, k_path_selected, initial_slot_selected, fail_info

    def valid_action_mask(self):
        request_size = self.current_VN_capacity.size
        node_capacities = self.current_observation["Vcap_Vbw_Scap"][2:]
        node_mask = self.generate_node_mask(request_size, node_capacities)
        self.node_mask = node_mask
        return node_mask


class VoneEnvRoutingOnly(VoneEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    (self.num_slots-self.min_slot_request+1) ** self.max_vnet_size,
                )
            )

    def select_nodes_paths_slots(self, action):
        """Get selected nodes, paths, and slots from action."""
        link_mapping_success = node_mapping_success = True
        fail_info = {}
        request_size = self.current_VN_capacity.size
        # Get node selection (dependent on number of nodes in request)
        nodes_selected, node_mapping_success = select_nodes_nsc(self, self.topology.topology_graph, self.current_observation)

        if self.routing_choose_k_paths:
            k_paths_selected = get_nth_item(self.generate_path_selection(request_size), action)

            link_mapping_success, k_paths_selected, initial_slots_selected = \
                select_path_fdl(
                    self,
                    self.topology.topology_graph,
                    self.current_VN_bandwidth,
                    nodes_selected,
                    k_paths_selected=k_paths_selected
                )

        else:
            k_paths_selected = get_nth_item(self.generate_path_selection(request_size), action[0])
            initial_slots_selected = get_nth_item(self.generate_slot_selection(request_size), action[1])

        if not node_mapping_success:
            fail_info = {"message": " Request failure: Node mapping by NSC method failed"}
        elif not link_mapping_success:
            fail_info = {"message": " Request failure: Link mapping failed"}

        logger.info(f' Nodes selected: {nodes_selected}')
        logger.info(f' Paths selected: {k_paths_selected}')
        logger.info(f' Slots selected: {initial_slots_selected}')

        return nodes_selected, k_paths_selected, initial_slots_selected, fail_info

    def valid_action_mask(self):
        # TODO - Try reframing action space to be discrete to allow masking
        request_size = self.current_VN_capacity.size
        path_mask = self.generate_path_mask(request_size)
        self.path_mask = path_mask
        return path_mask

    # TODO - Define new env for routing-only with masking,
    #  where the chosen nodes are presented as part of the observation


if __name__ == "__main___":
    # Test to check env is compatible with gym API
    print('Checking env')
    stable_baselines3.common.env_checker.check_env(VoneEnv)

    env = VoneEnv()
    print('Env created')
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
