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

import argparse


def train_part_2(local_config, teacher_save_path):
    project_name = local_config["project_name"]
    seed_val = local_config["seed_val"]
    lambda_val = local_config["lambda_val"]
    num_class = local_config["num_class"]

    meta_dir = local_config["meta_dir"]
    data_dir = local_config["data_dir"]
    dataset_name = local_config["dataset_name"]

    sweep_config = create_sweep_config(
        "Part_2",
    )

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def train():
        run = wandb.init()
        config = wandb.config

        # teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_teacher_output_seed_{config.seed}"
        teacher_ckpt_dir = find_best_checkpoint(teacher_save_path)

        command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/2.sh '2_{dataset_name.replace(' ', '_')}' {teacher_ckpt_dir} {lambda_val} {config.seed} {meta_dir} {data_dir} {num_class}"

        print(f"Executing: {command}")
        os.system(command)
        run.finish()

    wandb.agent(sweep_id, function=train, count=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "teacher_save_path", metavar="T", type=str, help="The teacher save folder path"
    )

    args = parser.parse_args()

    local_config_path = (
        "/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/FT_BERT.yml"
    )
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)

    # Ensure values are correctly loaded from the config file
    print(f"Loaded config: {local_config}")

    train_part_2(local_config, args.teacher_save_path)
