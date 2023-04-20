import networkx as nx
import numpy as np
import heapq

def rough_analytical_approximation(graph, traffic_load, C_v, S_v, avg_v_links = 3):
    """Calculate the average number of virtual networks that can be served

    Args:
        graph (networkx.classes.graph.Graph): NetworkX graph object
        traffic_load (int): Number of virtual networks to be served
        C_v (int): Virtual network capacity
        S_v (int): Number of slots per link

    Returns:
        int: Average number of virtual networks that can be served
    """
    # Calculate total physical capacity in nodes and links
    total_node_capacity = np.sum([graph.nodes[node]['capacity'] for node in graph.nodes()])
    total_link_capacity = np.sum([graph.edges[edge]['slots'] for edge in graph.edges()])

    # Calculate the required capacity for X virtual networks
    required_node_capacity = traffic_load * C_v
    required_link_capacity = traffic_load * avg_v_links * S_v

    # Calculate the number of virtual networks that can be served
    served_node_networks = min(total_node_capacity / required_node_capacity, 1) * traffic_load
    served_link_networks = min(total_link_capacity / required_link_capacity, 1) * traffic_load

    # Calculate the average number of virtual networks
    avg_virtual_networks = int(np.min(np.array([served_node_networks, served_link_networks])))

    return avg_virtual_networks


def blocking_probability(avg_virtual_networks_served, traffic_load):
    """Calculate the blocking probability

    Args:
        total_requests (int): Total number of requests
        avg_virtual_networks_served (int): Average number of virtual networks served
        traffic_load (int): Number of virtual networks requested to be served on average

    Returns:
        float: Blocking probability
    """

    blocking_prob = (traffic_load - avg_virtual_networks_served) / traffic_load
    return blocking_prob


def blocking_probability_simulation(graph, fixed_traffic_load, C_v, S_v, holding_time, total_requests):
    """Simulate the blocking probability

    Args:
        graph (networkx.classes.graph.Graph): NetworkX graph object
        fixed_traffic_load (int): Number of virtual networks to be served
        C_v (int): Virtual network capacity
        S_v (int): Number of slots per link
        holding_time (int): Holding time of the virtual network
        total_requests (int): Total number of requests

    Returns:
        float: Blocking probability
    """
    # Initialize the simulation
    current_time = 0
    num_requests = 0
    num_blocked = 0
    events = []
    active_requests = 0

    # Calculate the arrival rate based on the fixed traffic load and holding_time
    arrival_rate = fixed_traffic_load / holding_time

    # Generate initial arrival events
    inter_arrival_time = np.random.exponential(arrival_rate)
    heapq.heappush(events, (current_time + inter_arrival_time, 'arrival'))

    while events:
        # Get the next event
        current_time, event_type = heapq.heappop(events)

        # Stop the simulation if the simulation time is exceeded
        if num_requests == total_requests:
            break

        if event_type == 'arrival':
            num_requests += 1

            # Check if the request can be served
            avg_virtual_networks = rough_analytical_approximation(graph, fixed_traffic_load, C_v, S_v)
            if active_requests >= avg_virtual_networks:
                num_blocked += 1
            else:
                # Generate the departure event
                service_time = np.random.exponential(holding_time)
                heapq.heappush(events, (current_time + service_time, 'departure'))
                active_requests += 1

            # Generate the next arrival event
            inter_arrival_time = np.random.exponential(arrival_rate)
            heapq.heappush(events, (current_time + inter_arrival_time, 'arrival'))

        elif event_type == 'departure':
            active_requests -= 1

    return float(num_blocked / num_requests if num_requests > 0 else 0)


def calculate_nrtm_lrtm(graph, weight=None):
    """
    Calculate the Node Resource and Topology Metric (NRTM) and Link Resource and Topology Metric (LRTM).

    Parameters:
    -----------
    graph: networkx.Graph
        Input graph with 'capacity' and 'slots' attributes for nodes and edges, respectively.
    weight: str
        Edge attribute to use as weight for betweenness centrality calculations.

    Returns:
    --------
    NRTM: float
        Node Resource and Topology Metric
    LRTM: float
        Link Resource and Topology Metric
    """
    # Calculate node and link resource capacities
    node_resources = np.array(list(nx.get_node_attributes(graph, 'capacity').values()))
    link_resources = list(nx.get_edge_attributes(graph, 'slots').values())
    NRC = np.sum(node_resources)
    LRC = np.sum([len(slots) for slots in link_resources])


    # Calculate mean weighted node betweenness and mean edge betweenness
    node_betweenness = nx.betweenness_centrality(graph, weight=weight)
    edge_betweenness = nx.edge_betweenness_centrality(graph, weight=weight)
    MWNB = np.mean(list(node_betweenness.values()))
    MWEB = np.mean(list(edge_betweenness.values()))

    # Normalize the topological measures
    MWNB_range = np.max(list(node_betweenness.values())) - np.min(list(node_betweenness.values()))
    MWEB_range = np.max(list(edge_betweenness.values())) - np.min(list(edge_betweenness.values()))
    MWNB_norm = (MWNB - np.min(list(node_betweenness.values()))) / MWNB_range if MWNB_range != 0 else 1
    MWEB_norm = (MWEB - np.min(list(edge_betweenness.values()))) / MWEB_range if MWEB_range != 0 else 1

    # Node Resource and Topology Metric
    NRTM = MWNB_norm * NRC

    # Link Resource and Topology Metric
    LRTM = MWEB_norm * LRC

    return NRTM, LRTM


def create_virtual_network(adjacency_list, mean_node_request, mean_slot_request):
    G = nx.Graph()
    nodes = set([node for edge in adjacency_list for node in edge])
    G.add_nodes_from(nodes, capacity=mean_node_request)
    G.add_edges_from(adjacency_list, slots=np.full(mean_slot_request, 1))
    return G


def calculate_v_nrtm_lrtm(mean_node_request, mean_slot_request, adjacency_lists=[((0, 1), (1, 2), (0, 2))]):
    """
    Calculate the mean Node Resource and Topology Metric (NRTM) and Link Resource and Topology Metric (LRTM) for a set of virtual networks.

    Parameters:
    -----------
    mean_node_request: float
        Mean node resource request for the virtual networks.
    mean_slot_request: int
        Mean number of slots requested for the links in the virtual networks.
    adjacency_lists: list of tuples
        List of adjacency lists representing the virtual network topologies.

    Returns:
    --------
    mean_NRTM: float
        Mean Node Resource and Topology Metric for the virtual networks.
    mean_LRTM: float
        Mean Link Resource and Topology Metric for the virtual networks.
    """
    nrtm_values = []
    lrtm_values = []

    for adjacency_list in adjacency_lists:
        virtual_network = create_virtual_network(adjacency_list, mean_node_request, mean_slot_request)
        nrtm, lrtm = calculate_nrtm_lrtm(virtual_network)
        nrtm_values.append(nrtm)
        lrtm_values.append(lrtm)

    mean_NRTM = np.mean(nrtm_values)
    mean_LRTM = np.mean(lrtm_values)

    return mean_NRTM, mean_LRTM
