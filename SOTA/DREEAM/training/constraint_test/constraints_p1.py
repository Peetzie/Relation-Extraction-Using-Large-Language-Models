import os
import wandb
import yaml
from datetime import datetime
import glob


def create_sweep_config(
    part,
    seed_val,
    epochs,
    max_grad_norm,
    lambda_val,
    constraint_type="Constraints",  # Can be None or a list of possible values
    evidence_threshold=0.2,
    lr=5e-5,
):
    sweep_config = {
        "method": "grid",
        "metric": {"name": "rel_loss", "goal": "minimize"},
        "parameters": {
            "seed": {"values": seed_val},
            "epochs": {"values": epochs},
            "lambda_": {"values": lambda_val},
            "lr": {"values": lr},
            "max_grad_norm": {"values": max_grad_norm},
            "constraint_type": {"values": [constraint_type]},
            "evidence_threshold": {"values": evidence_threshold},
        },
    }
    return sweep_config


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


def run_training_pipeline(local_config):
    project_name = local_config["project_name"]
    lambda_val = local_config["lambda_val"]
    seed_val = local_config["seed_val"]
    num_class = local_config["num_class"]

    # Default hyperparameters for each part
    # Default hyperparameters for each part
    default_epochs_part_1 = 30
    default_lr_part_1 = 5e-5
    default_max_grad_norm_part_1 = 1.0

    default_epochs_part_3 = 2.0
    default_lr_part_3 = 3e-5
    default_max_grad_norm_part_3 = 5.0

    default_epochs_part_4 = 10
    default_lr_part_4 = 1e-6
    default_max_grad_norm_part_4 = 2.0

    for dataset in local_config["datasets"]:
        dataset_name = dataset["name"]
        constraint_types = ["constraints", "no_constraints"]

        for constraint in constraint_types:
            meta_dir = dataset[f"{constraint}_meta_dir"]
            data_dir = dataset[f"{constraint}_data_dir"]

            # Create sweep configurations for each part with the constraint type
            sweep_configs = {
                "Part_1": create_sweep_config(
                    part="Part_1",
                    seed_val=[seed_val],
                    epochs=[default_epochs_part_1],
                    max_grad_norm=[default_max_grad_norm_part_1],
                    lambda_val=[0.1],
                    constraint_type=constraint,  # Pass as list
                    evidence_threshold=[0.2],
                    lr=[default_lr_part_1],
                ),
            }

            sweep_ids = {
                part: wandb.sweep(config, project=project_name)
                for part, config in sweep_configs.items()
            }

            def train(part):
                run = wandb.init()
                config = wandb.config

                constraint_label = (
                    "Constraint"
                    if config.constraint_type == "constraints"
                    else "NoConstraint"
                )

                # Update the relevant paths
                teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_{constraint_label}_teacher_output_seed_{config.seed}"
                student_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_{constraint_label}_student_output_seed_{config.seed}"

                # Construct a dynamic name based on all hyperparameters
                hyperparameters = f"lambda{config.lambda_}_epochs{config.epochs}_lr{config.lr}_gradnorm{config.max_grad_norm}_seed{config.seed}"

                if part == "Part_1":
                    run = wandb.init(
                        project=project_name,
                        name=f"1_{dataset_name.replace(' ', '_')}_{constraint_label}_{hyperparameters}",
                        group=f"Part_1_{constraint_label}",
                        reinit=True,
                    )
                    command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/1.sh '1_{dataset_name.replace(' ', '_')}_{constraint_label}' {config.lambda_} {config.seed} {teacher_save_path} {meta_dir} {data_dir} {num_class} {config.epochs} {config.lr} {config.max_grad_norm}"
                print(f"Executing: {command}")
                os.system(command)
                run.finish()

            for part in ["Part_1"]:
                wandb.agent(sweep_ids[part], function=lambda: train(part))


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
