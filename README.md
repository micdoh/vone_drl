# Masked Deep Reinforcement Learning for Virtual Optical Network Embedding Over Elastic Optical Networks
___

Research project into the use of deep reinforcement learning (DRL) for optimisation of 
dynamic virtual optical network embedding (VONE) on an elastic optical network (EON) substrate.

Contains code for:

- [Gym](https://gymnasium.farama.org/) environments for VONE, 
with action spaces for node-and-path, node-only, and path-only selection.
- Scripts for: 
  - training
  - evaluation
  - data visualisation

Used to produce the results in [this paper](ONDM2023%20Masked%20DRL%20VONE.pdf), 
submitted to [ONDM 2023](https://ondm2023.inescc.pt/).


___
## How to reproduce ONDM 2023 paper results

### **Prerequisites:**

Activate virtual environment for project using `pyproject.toml` and [Poetry](https://python-poetry.org/) or `requirements.txt` and virtualenv/venv, as desired.

### **Instructions:**

The results shown in the paper can be recreated by:

1. Obtain trained models:
   - Option 1: Download pre-trained models
   - Option 2: Train from scratch
2. Evaluate model performance
3. Plot results

# Include command to run heuristic evaluation script

It's recommended to download the pre-trained models in order to avoid any difficulties with reproducing training 
(due to stochastic action sampling during training).

#### 1. (Optional) To train from scratch, run the following commands:

```commandline
poetry run python train.py --env_file ./config/agent_combined.yaml --log WARN --masking --recurrent_masking --gamma=0.5499732330963527 --learning_rate=0.001048322384752267 --n_steps=46 --output_file ./data/agent_nodes_train.csv --save_model --log_dir ./models --no_wandb --id combined --masking

poetry run python train.py --env_file ./config/agent_nodes.yaml --log WARN --masking --gamma=0.5664841514329136 --learning_rate=0.001309728909201273 --n_steps=63 --output_file ./data/agent_paths_train.csv --save_model --log_dir ./models --no_wandb --id nodes --masking

poetry run python train.py --env_file ./config/agent_paths.yaml --log WARN --masking --gamma=0.6344458270083639 --learning_rate=0.00031675064580752364 --n_steps=60 --output_file ./data/agent_combined_train.csv --save_model --log_dir ./models --no_wandb --id paths --masking --multistep_masking
```

#### 2. To evaluate models, run the following commands:

*To evaluate trained-from-scratch model:*

```commandline

poetry run python eval.py --env_file ./config/agent_combined.yaml --log WARN --model_file ./models/combined/combined_model.zip --output_file ./eval/agent_combined.csv --eval_masking --multistep_masking

poetry run python eval.py --env_file ./config/agent_nodes.yaml --log WARN --model_file ./models/nodes/nodes_model.zip --output_file ./eval/agent_nodes.csv --eval_masking

poetry run python eval.py --env_file ./config/agent_paths.yaml --log WARN --model_file ./models/paths/paths_model.zip --output_file ./eval/agent_paths.csv --eval_masking
```

*To evaluate pre-trained model:*
```commandline

poetry run python eval.py --env_file ./config/agent_combined.yaml --log WARN --output_file ./eval/agent_combined.csv --masking --artifact micdoh/VONE-DRL/agent_combined:v0

poetry run python eval.py --env_file ./config/agent_nodes.yaml --log WARN --output_file ./eval/agent_nodes.csv --masking --artifact micdoh/VONE-DRL/agent_nodes:v0

poetry run python eval.py --env_file ./config/agent_paths.yaml --log WARN --output_file ./eval/agent_paths.csv --masking --artifact micdoh/VONE-DRL/agent_paths:v0
```

*To generate heuristic evaluation data:*
```commandline
poetry run python eval_heuristic.py --output_file ./eval/heur.csv
```

#### 3. To create plots, run the following commands:

*For training curves:*
```commandline
poetry run python plot.py --training --node_train_file ./data/agent_nodes_train.csv --path_train_file ./data/agent_paths_train.csv --combined_train_file ./data/agent_combined_train.csv
```
*For evaluation results:*
```commandline
poetry run python plot.py --eval --node_eval_file ./eval/agent_nodes.csv --path_eval_file ./eval/agent_paths.csv --combined_eval_file ./eval/agent_combined.csv --heur_eval_file ./eval/heur.csv
```