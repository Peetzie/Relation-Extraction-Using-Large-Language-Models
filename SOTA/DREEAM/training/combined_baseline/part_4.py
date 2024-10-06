import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'finetune' folder
finetune_dir = os.path.join(current_dir, "../")

# Add 'finetune' folder to Python's module search path
sys.path.append(finetune_dir)

from common import create_sweep_config, find_best_checkpoint, load_config
import wandb


def train_part_4(local_config):
    project_name = local_config["project_name"]
    seed_val = local_config["seed_val"]
    lambda_val = local_config["lambda_val"]
    num_class = local_config["num_class"]

    default_epochs_part_4 = 10
    default_lr_part_4 = 1e-6
    default_max_grad_norm_part_4 = 2.0

    meta_dir = local_config["meta_dir"]
    data_dir = local_config["data_dir"]
    dataset_name = local_config["dataset_name"]

    sweep_config = create_sweep_config(
        "Part_4",
        seed_val,
        default_epochs_part_4,
        default_lr_part_4,
        default_max_grad_norm_part_4,
    )

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def train():
        run = wandb.init()
        config = wandb.config

        student_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/{dataset_name.replace(' ', '_')}_student_output_seed_{config.seed}"
        student_ckpt_dir = find_best_checkpoint(student_save_path)

        command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/4.sh '4_{dataset_name.replace(' ', '_')}' {student_ckpt_dir} {lambda_val} {config.seed} {student_save_path} {meta_dir} {data_dir} {num_class} {config.epochs} {config.lr} {config.max_grad_norm}"

        print(f"Executing: {command}")
        os.system(command)
        run.finish()

    wandb.agent(sweep_id, function=train, count=10)


if __name__ == "__main__":
    local_config = load_config()
    train_part_4(local_config)
