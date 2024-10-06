# common.py

import os
import wandb
import yaml
import glob


def create_sweep_config(
    part,
    seed_val=[66],
    epochs=[6],
    max_grad_norm=[1.0],
    lambda_val=[0.1],
    lr=[5e-5],
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
        },
    }
    return sweep_config


def find_best_checkpoint(path):
    search_pattern = os.path.join(path, "**", "best.ckpt")
    checkpoint_files = glob.glob(search_pattern, recursive=True)
    if checkpoint_files:
        return os.path.dirname(checkpoint_files[0])
    else:
        raise FileNotFoundError(f"best.ckpt not found in {path}")


def load_config(
    load_path="/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/testing.yml",
):
    local_config_path = load_path
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)
    return local_config
