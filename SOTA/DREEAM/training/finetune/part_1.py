import sys
import os

import yaml

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'finetune' folder
finetune_dir = os.path.join(current_dir, "../")

# Add 'finetune' folder to Python's module search path
sys.path.append(finetune_dir)

from common import create_sweep_config, find_best_checkpoint, load_config
import wandb


def train_part_1(local_config):
    project_name = local_config["project_name"]
    seed_val = local_config["seed_val"]
    num_class = local_config["num_class"]

    # epoch_list = [30.0, 50.0]
    # lr_list = [5e-4, 5e-5, 1e-5]
    # max_grad_norm_list = [0.5, 1.0, 1.5]
    # lambda_values = [0.01, 0.1, 0.3]

    # epoch_list = [30]
    # lr_list = [5e-5]
    # max_grad_norm_list = [1.0, 2.0]
    # lambda_values = [0.1]

    meta_dir = local_config["meta_dir"]
    data_dir = local_config["data_dir"]
    dataset_name = local_config["dataset_name"]

    # Create sweep configuration without constraints
    sweep_config = {
        "method": "bayes",  # Use Bayesian optimization
        "metric": {"name": "dev_f1", "goal": "maximize"},  # Optimize for dev_f1
        "early_terminate": {"type": "hyperband", "min_iter": 2},
        "parameters": {
            "epochs": {"min": 15, "max": 30},  # Range of epochs
            "lr": {"min": 1e-6, "max": 5e-4},  # Learning rate range
            "max_grad_norm": {"min": 0.5, "max": 2.0},  # Gradient norm range
            "lambda_": {"min": 0.1, "max": 0.3},  # Lambda values range
            "seed": {"values": [seed_val]},  # Seed for consistency
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def train():
        run = wandb.init()
        config = wandb.config

        # Construct the teacher save path
        teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_lambda_{config.lambda_}_epochs_{config.epochs}_lr_{config.lr}_gradnorm_{config.max_grad_norm}_teacher"

        # Construct the shell command
        command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/1.sh '1_{dataset_name.replace(' ', '_')}' {config.lambda_} {config.seed} {teacher_save_path} {meta_dir} {data_dir} {num_class} {config.epochs} {config.lr} {config.max_grad_norm}"

        print(f"Executing: {command}")
        os.system(command)
        run.finish()

    wandb.agent(sweep_id, function=train, count=14)


if __name__ == "__main__":
    local_config_path = (
        "/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/FT_BERT.yml"
    )
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)

    # Ensure values are correctly loaded from the config file
    print(f"Loaded config: {local_config}")

    train_part_1(local_config)
