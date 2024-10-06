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


def train_part_1(local_config):
    project_name = local_config["project_name"]
    seed_val = local_config["seed_val"]
    num_class = local_config["num_class"]

    epoch_list = [30, 36]
    lr_list = [1e-5, 3e-5, 5e-5]
    max_grad_norm_list = [1.0]
    lambda_values = [0.1]

    meta_dir = local_config["meta_dir"]
    data_dir = local_config["data_dir"]
    dataset_name = local_config["dataset_name"]

    # Create sweep configuration without constraints
    sweep_config = create_sweep_config(
        part="Part_1",
        seed_val=[seed_val],
        epochs=epoch_list,
        max_grad_norm=max_grad_norm_list,
        lambda_val=lambda_values,
        lr=lr_list,
    )

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def train():
        run = wandb.init()
        config = wandb.config

        # Construct the teacher save path
        teacher_save_path = f"/work3/s174159/LLM_Thesis/SOTA/DREEAM/wandb_out/{project_name.replace(' ', '_')}/teacher_output_seed_{config.seed}"

        # Construct the shell command
        command = f"sh /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/1.sh '1_{dataset_name.replace(' ', '_')}' {config.lambda_} {config.seed} {teacher_save_path} {meta_dir} {data_dir} {num_class} {config.epochs} {config.lr} {config.max_grad_norm}"

        print(f"Executing: {command}")
        os.system(command)
        run.finish()

    wandb.agent(sweep_id, function=train, count=2)


if __name__ == "__main__":
    local_config = load_config(
        load_path="/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/Nieller.yml"
    )
    train_part_1(local_config)
