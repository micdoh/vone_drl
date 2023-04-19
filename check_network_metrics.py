import gymnasium as gym
from heuristics import *
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import env.envs
import yaml
from estimate_bounds import calculate_nrtm_lrtm, calculate_v_nrtm_lrtm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", default="", type=str, help="Path to output csv file"
    )
    parser.add_argument(
        "--env_name", default="vone_Env-v0", type=str, help="Environment name"
    )
    parser.add_argument(
        "--episode_length", default=10000, type=int, help="Episode length"
    )
    parser.add_argument(
        "--load", default=60, type=int, help="Load"
    )
    parser.add_argument(
        "--mean_service_holding_time", default=10, type=int, help="Mean service holding time"
    )
    parser.add_argument(
        "--k_paths", default=5, type=int, help="Number of paths"
    )
    parser.add_argument(
        "--topology_name", default="nsfnet", type=str, help="Topology name"
    )
    parser.add_argument(
        "--num_slots", default=100, type=int, help="Number of slots"
    )
    parser.add_argument(
        "--node_capacity", default=30, type=int, help="Node capacity"
    )
    parser.add_argument(
        "--min_node_cap_request", default=1, type=int, help="Minimum node capacity request"
    )
    parser.add_argument(
        "--max_node_cap_request", default=2, type=int, help="Maximum node capacity request"
    )
    parser.add_argument(
        "--min_slot_request", default=2, type=int, help="Minimum slot request"
    )
    parser.add_argument(
        "--max_slot_request", default=4, type=int, help="Maximum slot request"
    )
    parser.add_argument(
        "--min_vnet_size", default=3, type=int, help="Minimum virtual network size"
    )
    parser.add_argument(
        "--max_vnet_size", default=3, type=int, help="Maximum virtual network size"
    )
    parser.add_argument(
        "--node_heuristic", default="nsc", type=str, help="Node heuristic"
    )
    parser.add_argument(
        "--path_heuristic", default="ff", type=str, help="Path heuristic"
    )
    parser.add_argument(
        "--num_episodes", default=1, type=int, help="Number of episodes"
    )
    args = parser.parse_args()

    env_args = {
    "episode_length": args.epispde_length,
    "load": args.load,
    "mean_service_holding_time": args.mean_service_holding_time,
    "k_paths": args.k_paths,
    "topology_name": args.topology_name,
    "num_slots": args.num_slots,
    "node_capacity": args.node_capacity,
    "min_node_cap_request": args.min_node_cap_request,
    "max_node_cap_request": args.max_node_cap_request,
    "min_slot_request": args.min_slot_request,
    "max_slot_request": args.max_slot_request,
    "min_vnet_size": args.min_vnet_size,
    "max_vnet_size": args.max_vnet_size,
    "vnet_size_dist": "fixed",
    }

    data_file = Path(args.output_file)
    data_file.parent.mkdir(exist_ok=True)

    for i in range(0, 20):

        mean_v_cap_request = env_args["min_node_cap_request"] + (
                env_args["max_node_cap_request"] - env_args["min_node_cap_request"]) / 2
        mean_v_slot_request = env_args["min_slot_request"] + (
                env_args["max_slot_request"] - env_args["min_slot_request"]) / 2

        the_env = gym.make(args.env_name, **env_args)

        # Calc NRTM and LRTM
        nrtm, lrtm = calculate_nrtm_lrtm(the_env.topology.topology_graph)
        v_nrtm, v_lrtm = calculate_v_nrtm_lrtm(mean_v_cap_request, mean_v_slot_request)

        results = []
        for ep in range(args.num_episodes):

            obs = the_env.reset()
            result, timestep_info_df = run_heuristic(
                the_env,
                node_heuristic=args.node_heuristic,
                path_heuristic=args.path_heuristic,
            )
            result["nrtm"] = nrtm
            result["lrtm"] = lrtm
            result["v_nrtm"] = v_nrtm
            result["v_lrtm"] = v_lrtm
            result["nrtm_ratio"] = nrtm / v_nrtm
            result["lrtm_ratio"] = lrtm / v_lrtm
            results.append(result)

        # Increase the mean request size
        env_args["min_node_cap_request"] += 1
        env_args["max_node_cap_request"] += 1
        env_args["min_slot_request"] += 1
        env_args["max_slot_request"] += 1

        df = pd.DataFrame(results)
        # Getting the mean reward and mean standard deviation of reward per episode
        df = pd.DataFrame([df.mean().to_dict()])
        df.to_csv(data_file, mode='a', header=not os.path.exists(data_file))
        print(f"ep {ep} done")

    the_env.close()
