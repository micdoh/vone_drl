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


def blocking_probability_simulation(graph, fixed_traffic_load, C_v, S, holding_time, total_requests):
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
            avg_virtual_networks = rough_analytical_approximation(graph, fixed_traffic_load, C_v, S)
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

    return num_blocked / num_requests if num_requests > 0 else 0


