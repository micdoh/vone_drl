import gymnasium as gym
import numpy as np
import os
from pathlib import Path
from itertools import islice
from typing import Generator, Any, Callable
import logging
import argparse
import time
import json
import networkx as nx
import re
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--env_file", type=str, help="Absolute path to config file for run"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable weights and biases logging",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for optimisation",
    )
    parser.add_argument(
        "--gamma", default=0.99, type=float, help="Discount factor"
    )
    parser.add_argument(
        "--clip_range",
        default=0.2,
        type=float,
        help="Clipping range for PPO",
    )
    parser.add_argument(
        "--clip_range_vf",
        default=None,
        type=float,
        help="Clipping range for value function",
    )
    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
        help="No. of times to update ",
    )
    parser.add_argument(
        "--ent_coef",
        default=0.0,
        type=float,
        help="Coefficient for entropy term in the loss function",
    )
    parser.add_argument(
        "--gae_lambda",
        default=0.908,
        type=float,
        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator",
    )
    parser.add_argument(
        "--reward_success",
        default=10,
        type=int,
        help="Reward for successful request",
    )
    parser.add_argument(
        "--reward_failure",
        default=0,
        type=int,
        help="Reward for failed request",
    )
    parser.add_argument(
        "--n_steps",
        default=50,
        type=int,
        help="The number of steps to run for each environment per update "
        "(i.e. rollout buffer size is n_steps * n_envs where n_envs "
        "is number of environment copies running in parallel)",
    )
    parser.add_argument(
        "--reject_action",
        action="store_true",
        help="Bool to use reject action",
    )
    parser.add_argument(
        "--lr_schedule",
        default=None,
        type=str,
        help="Type of learning rate schedule to use",
    )
    parser.add_argument(
        "--clip_range_schedule",
        default=None,
        type=str,
        help="Type of clipping range schedule to use",
    )
    parser.add_argument(
        "--clip_range_vf_schedule",
        default=None,
        type=str,
        help="Type of clipping range schedule to use for value function",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="No. of samples per batch"
    )
    parser.add_argument(
        "--n_procs",
        default=1,
        type=int,
        help="No. of processes to run parallel environments",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Bool to use parallel processes with vectorised environments",
    )
    parser.add_argument("--log", default="WARN", type=str, help="Set log level")
    parser.add_argument(
        "--masking", action="store_true", help="Use invalid action masking"
    )
    parser.add_argument(
        "--multistep_masking", action="store_true", help="Use multistep invalid action masking"
    )
    parser.add_argument(
        "--multistep_masking_attr",
        default="curr_selection",
        type=str,
        help="Specify environment variables to be accessed sequentially in multistep masking",
    )
    parser.add_argument(
        "--multistep_masking_n_steps",
        default=3,
        type=int,
        help="Specify number of steps to mask in multistep masking",
    )
    parser.add_argument(
        "--action_interpreter",
        default="select_nodes_paths_slots",
        help="Specify function that interprets actions in the env. To be used in multistep action masking.",
    )
    parser.add_argument(
        "--eval_masking", action="store_true", help="Eval: Use invalid action masking for evaluation"
    )
    parser.add_argument(
        "--eval_multistep_masking", action="store_true", help="Eval: Use multistep invalid action masking for evaluation"
    )
    parser.add_argument(
        "--output_file", default="", type=str, help="Eval: Path to output csv file"
    )
    parser.add_argument(
        "--min_load",
        default=1,
        type=int,
        help="Eval: Min. of range of loads to evaluate",
    )
    parser.add_argument(
        "--max_load",
        default=15,
        type=int,
        help="Eval: Max. of range of loads to evaluate",
    )
    parser.add_argument(
        "--load_step", default=1, type=int, help="Eval: Step size of loads to evaluate"
    )
    parser.add_argument(
        "--model_file", default="", type=str, help="Path to model zip file for retraining"
    )
    parser.add_argument(
        "--artifact", default="", type=str, help="Name of artiact to download from wandb"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Use callback to save model with best training reward",
    )
    parser.add_argument(
        "--policy_type",
        default="MultiInputPolicy",
        help="Specify type of policy to be used by agent",
    )
    parser.add_argument(
        "--total_timesteps",
        default=135000,
        type=int,
        help="Timesteps to run training for",
    )
    parser.add_argument(
        "-i", "--id", type=str, help="ID to name this evaluation run"
    )
    parser.add_argument(
        "--save_episode_info",
        action="store_true",
        help="Save episode rewards to file",
    )
    parser.add_argument(
        "--log_dir",
        default="./runs",
        help="Parent directory for all log directories",
    )
    parser.add_argument(
        "--use_afterstate",
        type=bool, default=False,
        help="Use afterstate as input to value function",
    )
    parser.add_argument(
        "--n_eval_episodes", default=2, type=int, help="Eval: Number of episodes to evaluate at each load"
    )
    parser.add_argument(
        "--callback", action="store_true", help="Use callback to save model and/or data"
    )
    parser.add_argument(
        "--topology_name", default="nsfnet", type=str, help="Name of topology"
    )
    parser.add_argument(
        "--topology_file", default="", type=str, help="Path to topology file"
    )
    parser.add_argument(
        "--num_nodes", default=14, type=int, help="Number of nodes in topology"
    )
    parser.add_argument(
        "--connectivity", default=0.15, type=float, help="Connectivity of topology"
    )
    parser.add_argument(
        "--ws_rewire_prob", default=0.1, type=float, help="Rewiring probability of topology"
    )
    parser.add_argument(
        "--num_layers", default=2, type=int, help="Number of layers in policy network",
    )
    parser.add_argument(
        "--num_hidden", default=64, type=int, help="Number of hidden units in policy network",
    )
    parser.add_argument(
        "--custom_vf", action="store_true", help="Use custom value function"
    )
    parser.add_argument(
        "--num_layers_vf", default=2, type=int, help="Number of layers in value function network",
    )
    parser.add_argument(
        "--num_hidden_vf", default=64, type=int, help="Number of hidden units in value function network",
    )
    parser.add_argument(
        "--decrease_min_slot_request", action="store_true", help="Decrease min slot request over time"
    )
    parser.add_argument(
        "--step_min_slot_request", default=1, type=float, help="Step size for decreasing min slot request"
    )
    parser.add_argument(
        "--ep_step_min_slot_request", default=10, type=int, help="Episode step size for decreasing min slot request"
    )
    return parser.parse_args()



def init_logger(log_file: str, loglevel: str):
    logger = logging.getLogger()
    loglevel = getattr(logging, loglevel.upper())
    logger.setLevel(loglevel)
    # create file handler which logs event debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(loglevel)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def choose_schedule(schedule: str, initial_value: float):
    """Choose schedule for learning rate"""
    if schedule is None or "constant":
        return initial_value
    elif schedule == "linear":
        return linear_schedule(initial_value)
    else:
        raise ValueError("Schedule not recognised")


def define_logs(run_id, log_dir, loglevel):
    log_dir = Path(log_dir) / run_id
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    logger = init_logger(log_file.absolute(), loglevel)
    model_save_file = log_dir / f"{run_id}_model.zip"
    return (
        log_dir,
        log_file,
        logger,
        model_save_file,
    )


def get_nth_item(gen: Generator, n: int) -> Any:
    """Return the nth item from a generator"""
    return next(islice(gen, n, None), None)


def get_gen_index(gen: Generator, value: [int]) -> int:
    """Get index of generator value that matches value.

    Used to translate from requirement value back to request integer,
    which is the index of the matching table row.

    Args:
        gen: Generator to retrieve value-index from.
        value: list of values to match to generator
    """
    for n, item in enumerate(gen):
        if (item == np.array(value)).all():
            return n


def mask_fn(env: gym.Env) -> np.ndarray:
    """Do whatever you'd like in this function to return the action mask for the env.
    In this example, we assume the env has a helpful method we can rely on.
    Don't forget to use MaskablePPO and ActionMasker wrapper for the env."""
    return env.valid_action_mask()


def make_env(env_id, seed, **kwargs):
    """Used for instantiating multiple (vectorised) envs.
    Disable env checker to ensure env is not wrapped (to set env attrs for action masking)"""
    def _init():
        env = gym.make(env_id, disable_env_checker=True, **kwargs)
        env.seed(seed)
        return env

    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f'func:{f.__name__} args:[{args}, {kw}] took:{te-ts:2.4f} sec')
        return result

    return timed


def create_conus(connections, locations=False):
    G = nx.Graph()

    # Helper function to extract city name from a node
    def extract_city(node_name):
        """
        Given a node name, extracts all city names from the node name using regex.
        """
        matches = re.findall(r'[A-Z][a-z_]+\s*[A-Za-z_]+', node_name)
        return matches if matches else None

    # First pass: Create unique city nodes
    city_nodes = set()
    for connection in connections:
        from_node = connection['from_node']
        to_node = connection['to_node']

        # Extract city names from nodes
        from_city = extract_city(from_node)
        to_city = extract_city(to_node)

        if from_city:
            for city in from_city:
                city_nodes.add(city)
        if to_city:
            for city in to_city:
                city_nodes.add(city)

    # Sort city nodes alphabetically
    city_nodes = sorted(list(city_nodes))

    if locations:
        # Add unique city nodes to the graph with locations
        geolocator = Nominatim(user_agent="geoapiExercises")
        city_coords = {city: (geolocator.geocode(city + ', USA').latitude, geolocator.geocode(city + ', USA').longitude) for
                       city in city_nodes}
        for city, coords in city_coords.items():
            G.add_node(city, pos=coords)
    else:
        for city in city_nodes:
            G.add_node(city)

    # Collect unique edges in a set
    unique_edges = set()
    for connection in connections:
        from_node = connection['from_node']
        to_node = connection['to_node']

        from_cities = extract_city(from_node)
        to_cities = extract_city(to_node)

        # Add an edge between the city nodes if both cities are present and the edge doesn't exist
        for from_city in from_cities:
            for to_city in to_cities:
                if from_city == to_city:
                    continue
                if from_city and to_city:
                    edge = tuple(sorted((from_city, to_city)))
                    unique_edges.add(edge)

    # Add unique edges to the graph
    for edge in unique_edges:
        G.add_edge(*edge)

    # Remove underscores from names
    G = nx.relabel_nodes(G, {node: node.replace('_', ' ') for node in G.nodes})

    return G


def load_conus_topology(path, plot=False):

    with open(path, 'r') as file:
        conus_data = json.load(file)
    G = create_conus(conus_data["connections"], locations=plot)

    if plot:
        G = nx.convert_node_labels_to_integers(G, label_attribute="name")
        # Get coordinates from the graph
        pos = nx.get_node_attributes(G, 'pos')

        # Create a GeoDataFrame with the city coordinates
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([lon for lat, lon in pos.values()], [lat for lat, lon in pos.values()]),
            index=pos.keys())
        gdf.crs = 'EPSG:4326'
        gdf = gdf.to_crs('EPSG:3857')

        # Create the plot using Seaborn, Matplotlib, and Contextily
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the nodes and edges
        for edge in G.edges:
            edge_coords = [gdf.loc[edge[0], 'geometry'], gdf.loc[edge[1], 'geometry']]
            ax.plot([edge_coords[0].x, edge_coords[1].x], [edge_coords[0].y, edge_coords[1].y], color='darkblue', linewidth=5, zorder=1)

        ax.scatter(gdf.geometry.x, gdf.geometry.y, facecolors='white', edgecolors='blue', linewidths=2, s=400, zorder=2)

        # Add node labels
        for node, attributes in G.nodes(data=True):
            coord = gdf.loc[node, 'geometry']
            ax.text(coord.x, coord.y, node, ha='center', va='center', fontsize=15, zorder=3)

        # Remove the grid, legend, and axis labels
        ax.grid(False)
        ax.legend().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the aspect ratio to make the map look correct
        ax.set_aspect('equal', adjustable='box')

        # Save the plot to a file
        plt.savefig("conus_topology.png", dpi=300, bbox_inches='tight', pad_inches=0)

        # Show the plot
        plt.show()

    return G


if __name__ == "__main__":
    conus = load_conus_topology('./topologies/CORONET_CONUS_Topology.json', plot=True)
    for node, attrs in conus.nodes(data=True):
        print(f"Node: {node}, Attributes: {attrs}")
        # Print edge data
        print(f"Edges: {conus.edges(node, data=True)}")
    print(conus)
