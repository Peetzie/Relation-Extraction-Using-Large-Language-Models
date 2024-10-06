import os
import subprocess
import yaml
import argparse


def run_evaluation(
    name, model_dir, split, num_class, data_dir, meta_dir, predictions_output
):
    # Print the command for logging/debugging
    print(f"Running evaluation with the following parameters:")
    print(f"NAME={name}")
    print(f"MODEL_DIR={model_dir}")
    print(f"SPLIT={split}")
    print(f"NUM_CLASS={num_class}")
    print(f"DATA_DIR={data_dir}")
    print(f"META_DIR={meta_dir}")

    # Base command for running the evaluation
    base_command = (
        f"python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py "
        f'--data_dir "{data_dir}" '
        f'--meta_dir "{meta_dir}" '
        f"--transformer_type bert "
        f"--model_name_or_path bert-base-cased "
        f'--display_name "{name}" '
        f'--load_path "{model_dir}" '
        f'--test_file "{split}.json" '
        f"--test_batch_size 8 "
        f"--num_labels 4 "
        f"--evi_thresh 0.2 "
        f'--num_class "{num_class}" '
        f"--do_test "
        f"--predictions_output {predictions_output}"
    )

    # Run the command in single evaluation mode
    single_eval_command = base_command + " --eval_mode single"
    print(f"Executing: {single_eval_command}")
    subprocess.run(single_eval_command, shell=True)

    # Run the command in fusion evaluation mode
    fusion_eval_command = base_command + " --eval_mode fushion"
    print(f"Executing: {fusion_eval_command}")
    subprocess.run(fusion_eval_command, shell=True)


if __name__ == "__main__":
    # Load the local configuration from the YAML file in the config/baseline subfolder
    local_config_path = "/work3/s174159/LLM_Thesis/SOTA/DREEAM/training/config/Roberta_Combined_Default.yml"
    with open(local_config_path) as config_file:
        local_config = yaml.safe_load(config_file)

    # Ensure values are correctly loaded from the config file
    print(f"Loaded config: {local_config}")

    # Retrieve values from the config file
    project_name = local_config["project_name"]
    num_class = local_config["num_class"]
    data_dir = local_config["data_dir"]  # Assuming you're using the constraints data
    meta_dir = local_config["meta_dir"]
    predictions_output = local_config["predictions_output"]

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation for a specific model directory and data split."
    )
    parser.add_argument(
        "model_dir", type=str, help="The directory containing the trained model"
    )
    parser.add_argument(
        "split", type=str, help="The data split to evaluate (e.g., dev)"
    )

    args = parser.parse_args()

    # Run the evaluation with the loaded local configuration
    run_evaluation(
        project_name,
        args.model_dir,
        args.split,
        num_class,
        data_dir,
        meta_dir,
        predictions_output,
    )
