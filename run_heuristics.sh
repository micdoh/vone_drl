#!/bin/bash

# Define the lists of parameters
PH=(ff fdl msp_ef)
NH=(nsc calrc tmr)

# Loop over all combinations of PH and NH and run the command
for p in ${PH[@]}; do
  for n in ${NH[@]}; do
    # Replace PH and NH in the command string with the current values of p and n
    cmd="python3 eval_heuristic.py --env_file ./config/agent_conus.yaml --output_file ./eval/${n}_${p}_conus.csv --node_heuristic $n --path_heuristic $p --min_load 80 --max_load 100 --load_step 10 --num_episodes 1"
    echo "Running command: $cmd"
    eval $cmd
  done
done