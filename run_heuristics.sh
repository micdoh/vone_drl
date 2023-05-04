#!/bin/bash

# Define the lists of parameters
PH=(ff fdl msp_ef)
NH=(nsc calrc tmr)

# Loop over all combinations of PH and NH and run the command
for p in ${PH[@]}; do
  for n in ${NH[@]}; do
    # Replace PH and NH in the command string with the current values of p and n
    cmd="python3 eval_heuristic.py --env_file ./config/agent_combined.yaml --output_file ./eval/${n}_${p}_100slots.csv --node_heuristic $n --path_heuristic $p --min_load 40 --max_load 100 --load_step 1 --num_episodes 3"
    echo "Running command: $cmd"
    eval $cmd
  done
done