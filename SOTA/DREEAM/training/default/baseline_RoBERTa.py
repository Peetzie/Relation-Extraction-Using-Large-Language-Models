import os
import wandb
import yaml
from datetime import datetime
import glob


def create_sweep_config(part, seed_val=66, epochs=30, lr=5e-5, max_grad_norm=1.0):
    return {
        "method": "grid",
        "metric": {"name": "rel_loss", "goal": "minimize"},
        "parameters": {
            "seed": {"values": [seed_val]},
            "epochs": {"values": [epochs]},
            "lr": {"values": [lr]},
            "max_grad_norm": {"values": [max_grad_norm]},
        },
    }


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
    dataset_name = local_config["dataset_name"]
    project_name = local_config["project_name"]
    lambda_val = local_config["lambda_val"]
    seed_val = local_config["seed_val"]
    num_class = local_config["num_class"]
    meta_dir = local_config["meta_dir"]
    data_dir = local_config["data_dir"]

    # Default hyperparameters for each part
    default_epochs_part_1 = 30
    default_lr_part_1 = 3e-5
    default_max_grad_norm_part_1 = 1.0

    default_epochs_part_3 = 5.0
    default_lr_part_3 = 1e-5
    default_max_grad_norm_part_3 = 5.0

    default_epochs_part_4 = 10
    default_lr_part_4 = 1e-5
    default_max_grad_norm_part_4 = 5.0

    # Debug prints to verify the values
    print(f"dataset_name: {dataset_name}")
    print(f"project_name: {project_name}")
    print(f"lambda_val: {lambda_val}")
    print(f"seed_val: {seed_val}")
    print(f"num_class: {num_class}")
    print(f"meta_dir: {meta_dir}")
    print(f"data_dir: {data_dir}")

    # Create sweep configurations for each part
    sweep_configs = {
        "Part_1": create_sweep_config(
            "Part_1",
            seed_val,
            default_epochs_part_1,
            default_lr_part_1,
            default_max_grad_norm_part_1,
        ),
        "Part_2": create_sweep_config("Part_2", seed_val, None, None, None),
        "Part_3": create_sweep_config(
            "Part_3",
            seed_val,
            default_epochs_part_3,
            default_lr_part_3,
            default_max_grad_norm_part_3,
        ),
        "Part_4": create_sweep_config(
            "Part_4",
            seed_val,
            default_epochs_part_4,
            default_lr_part_4,
            default_max_grad_norm_part_4,
        ),
    }

    sweep_ids = {
        part: wandb.sweep(config, project=project_name)
        for part, config in sweep_configs.items()
    }

    def train(part):
        run = wandb.init()
        config = wandb.config

        # Ensure values are correctly passed
        lambda_val = local_config["lambda_val"]
        seed_val = local_config["seed_val"]
        num_class = local_config["num_class"]
        meta_dir = local_config["meta_dir"]
        data_dir = local_config["data_dir"]
        epochs = config.get("epochs")
        lr = config.get("lr")
        max_grad_norm = config.get("max_grad_norm")

        # Debug prints to verify the values during training
        print(f"lambda_val during training: {lambda_val}")
        print(f"seed_val during training: {seed_val}")
        print(f"num_class during training: {num_class}")
        print(f"meta_dir during training: {meta_dir}")
        print(f"data_dir during training: {data_dir}")
        print(f"epochs during training: {epochs}")
        print(f"lr during training: {lr}")
        print(f"max_grad_norm during training: {max_grad_norm}")

        if part == "Part_1":
            # Step 1: Training the teacher model
            teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/teacher_output_seed_{config.seed}"
            run = wandb.init(
                project=project_name,
                name=f"1_{dataset_name.replace(' ', '_')}_Constraints_lambda{lambda_val}_{config.seed}",
                group="Part_1",
                reinit=True,
            )
            command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/1.sh '1_{dataset_name.replace(' ', '_')}_Constraints' {lambda_val} {config.seed} {teacher_save_path} {meta_dir} {data_dir} {num_class} {epochs} {lr} {max_grad_norm}"
        elif part == "Part_2":
            # Step 2: Infer on distantly-supervised data
            teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/teacher_output_seed_{config.seed}"
            teacher_ckpt_dir = find_best_checkpoint(teacher_save_path)
            run = wandb.init(
                project=project_name,
                name=f"2_{dataset_name.replace(' ', '_')}_Infer_lambda{lambda_val}_{config.seed}",
                group="Part_2",
                reinit=True,
            )
            command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/2.sh '2_{dataset_name.replace(' ', '_')}_Infer' {teacher_ckpt_dir} {lambda_val} {config.seed} {meta_dir} {data_dir} {num_class}"
        elif part == "Part_3":
            # Step 3: Self-training on distantly-supervised data
            teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/teacher_output_seed_{config.seed}"
            teacher_ckpt_dir = find_best_checkpoint(teacher_save_path)
            student_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/student_output_seed_{config.seed}"
            run = wandb.init(
                project=project_name,
                name=f"3_{dataset_name.replace(' ', '_')}_SelfTrain_lambda{lambda_val}_{config.seed}",
                group="Part_3",
                reinit=True,
            )
            command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/3RoBERTa.sh '3_{dataset_name.replace(' ', '_')}_SelfTrain' {teacher_ckpt_dir} {lambda_val} {config.seed} {student_save_path} {meta_dir} {data_dir} {num_class} {epochs} {lr} {max_grad_norm}"
        elif part == "Part_4":
            # Step 4: Fine-tuning on human-annotated data
            student_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/student_output_seed_{config.seed}"
            student_ckpt_dir = find_best_checkpoint(student_save_path)
            run = wandb.init(
                project=project_name,
                name=f"4_{dataset_name.replace(' ', '_')}_FineTune_lambda{lambda_val}_{config.seed}",
                group="Part_4",
                reinit=True,
            )
            command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/4RoBERTa.sh '4_{dataset_name.replace(' ', '_')}_FineTune' {student_ckpt_dir} {lambda_val} {config.seed} {student_save_path} {meta_dir} {data_dir} {num_class} {epochs} {lr} {max_grad_norm}"

        print(f"Executing: {command}")
        os.system(command)
        run.finish()

    for part in ["Part_1", "Part_2", "Part_3", "Part_4"]:
        wandb.agent(sweep_ids[part], function=lambda: train(part), count=10)


if __name__ == "__main__":
    # Load the local configuration from the YAML file in the config/baseline subfolder
    local_config_path = (
        "/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/default_RoBERTa.yml"
    )
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)

    # Ensure values are correctly loaded from the config file
    print(f"Loaded config: {local_config}")

    # Run the training pipeline with the loaded local configuration
    run_training_pipeline(local_config)
