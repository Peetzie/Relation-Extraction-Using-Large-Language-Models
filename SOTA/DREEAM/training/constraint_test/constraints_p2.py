import os
import wandb
import yaml
from datetime import datetime
import glob


def find_best_checkpoint(path):
    # Search for best.ckpt in the provided path and its subdirectories
    search_pattern = os.path.join(path, "**", "best.ckpt")
    checkpoint_files = glob.glob(search_pattern, recursive=True)
    if checkpoint_files:
        return os.path.dirname(
            checkpoint_files[0]
        )  # Return the directory containing best.ckpt
    else:
        raise FileNotFoundError(f"best.ckpt not found in {path}")


def create_sweep_config(seed_val, lambda_val, constraint_type):
    sweep_config = {
        "method": "grid",
        "parameters": {
            "seed": {"values": [seed_val]},
            "lambda_": {"values": [lambda_val]},
            "constraint_type": {"values": [constraint_type]},
        },
    }
    return sweep_config


def run_training_pipeline(local_config):
    project_name = local_config["project_name"]
    lambda_val = local_config["lambda_val"]
    seed_val = local_config["seed_val"]
    num_class = local_config["num_class"]

    constraint_types = ["constraints", "no_constraints"]

    for constraint in constraint_types:  # Create a sweep for each constraint type
        sweep_config = create_sweep_config(seed_val, lambda_val, constraint)
        sweep_id = wandb.sweep(sweep_config, project=project_name)

        def train(config=None):
            with wandb.init(config=config) as run:
                config = wandb.config

                constraint_label = (
                    "Constraint"
                    if config.constraint_type == "constraints"
                    else "NoConstraint"
                )

                for dataset in local_config["datasets"]:
                    dataset_name = dataset["name"]
                    meta_dir = dataset[f"{constraint}_meta_dir"]
                    data_dir = dataset[f"{constraint}_data_dir"]

                    # Load the checkpoint from Part 1
                    teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_{constraint_label}_teacher_output_seed_{config.seed}"
                    teacher_ckpt_dir = find_best_checkpoint(teacher_save_path)

                    # Construct the training command
                    command = (
                        f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/2.sh "
                        f"'2_{dataset_name.replace(' ', '_')}_{constraint_label}_Infer' {teacher_ckpt_dir} {lambda_val} "
                        f"{config.seed} {meta_dir} {data_dir} {num_class}"
                    )

                    print(f"Executing: {command}")
                    os.system(command)

        # Use the sweep ID in the wandb.agent call
        wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    # Load the local configuration from the YAML file in the config/baseline subfolder
    local_config_path = (
        "/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/constraint.yml"
    )
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)

    # Ensure values are correctly loaded from the config file
    print(f"Loaded config: {local_config}")

    # Run the training pipeline with the loaded local configuration
    run_training_pipeline(local_config)
