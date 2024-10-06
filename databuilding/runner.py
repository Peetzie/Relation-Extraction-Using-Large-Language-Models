import argparse
import importlib.util
import sys
from pathlib import Path
import inquirer
from colorama import Fore, Style, init
import glob
import os
import shutil

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the 'finetune' folder
scripts_DIR = os.path.join(current_dir, "components", "scripts")

# Add 'finetune' folder to Python's module search path
sys.path.append(scripts_DIR)

from download_utils import *


# Map dataset names to script paths
SCRIPTS = {
    "CoNLL04": "components/scripts/CoNLL04_prepare.py",
    "CrossRE": "components/scripts/CrossRE_prepare.py",
    "DocRED(Distant)": "components/scripts/DocRED_distant_prepare.py",
    "DocRED (Normal / Joint)": "components/scripts/DocRED_joint_prepare.py",
    "ReDocRED": "components/scripts/ReDocRED_prepare.py",  # Added ReDocRED
    "NYT": "components/scripts/NYT_prepare.py",
    "SciERC": "components/scripts/SciERC_prepare.py",
    "REBEL (Requires approx 90GB Ram)": "components/scripts/REBEL_prepare.py",
    # "ADE": "components/scripts/ADE_prepare.py",
    # "WebNLG": "components/scripts/WebNLG_prepare.py",
}


def load_and_run_module(
    script_path, args, use_file_no, combine=False, JSON=False, verify=False, split=False
):
    module_name = Path(script_path).stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if combine:
        module.main(args.folders, args.output_dir, args.verbose)
    elif JSON:
        module.main(args.input_file, args.output_dir)
    elif verify:
        print(args.input_file, args.output_dir, args.model_name, args.max_seq_length)
        module.main(
            args.input_file, args.output_dir, args.model_name, args.max_seq_length
        )
    elif split:
        module.main(
            args.input_file, args.output_dir, args.dataset_name, args.gen_new_distant
        )
    else:
        if use_file_no:
            if args.dataset == "REBEL (Requires approx 90GB Ram)":
                module.main(
                    args.data_dir,
                    args.output_dir,
                    args.download,
                    True,
                    args.parse_entities,
                )
            else:
                module.main(
                    args.data_dir,
                    args.output_dir,
                    args.download,
                    True,
                )
        else:
            module.main(args.data_dir, args.output_dir, args.download, True)


def get_available_datasets(data_dir, default_data_dir):
    if data_dir == default_data_dir:
        # Look for folders containing "_Modified" in the default directory
        folders = glob.glob(str(Path(data_dir, "*_Modified")))
    else:
        # List the names of folders in the specified directory by the user
        folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    available_datasets = [
        Path(folder).stem.replace("_Modified", "") for folder in folders
    ]
    return available_datasets


def scan_preprocessed_data(preprocessed_dir):
    datasets = [f.name for f in os.scandir(preprocessed_dir) if f.is_dir()]
    datasets.append("ALL (COMBINED)")
    return datasets


def scan_corrected_files(preprocessed_dir):
    corrected_files = glob.glob(
        str(Path(preprocessed_dir, "**/*_corrected.json")), recursive=True
    )
    datasets = set(Path(file).parent.name for file in corrected_files)
    if Path(preprocessed_dir, "combined_dataset_corrected.json").exists():
        datasets.add("ALL (COMBINED)")
    return datasets


def main():
    while True:
        # Initial question to determine the action
        action_question = [
            inquirer.List(
                "action",
                message="Select an action:",
                choices=[
                    "1. Download",
                    "2. Combine",
                    "3. Export JSON types",
                    "4. Clean",
                    "5. Verify",
                    "6. Split",
                ],
            ),
        ]

        action_answer = inquirer.prompt(action_question)

        default_data_dir = f"{root_folder}/data/raw_data"
        default_preprocessed_dir = f"{root_folder}/data/pre-processed data"
        default_finalized_dir = f"{root_folder}/data/data_finalized"

        if action_answer["action"] == "1. Download":
            # Ask for dataset selection if downloading individually
            dataset_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to download",
                    choices=SCRIPTS.keys(),
                ),
            ]
            questions = dataset_question + [
                inquirer.Text(
                    "data_dir",
                    message="Location to download the data files. Leave blank for default",
                    default=default_data_dir,
                ),
                inquirer.Text(
                    "output_dir",
                    message="Output directory to save results. Leave blank for default",
                    default=default_preprocessed_dir,
                ),
                inquirer.Confirm(
                    "download", message="Download the data files?", default=True
                ),
            ]
            answers = inquirer.prompt(questions)

            for dataset in answers["datasets"]:
                # Create an argparse.Namespace object to hold the arguments
                args = argparse.Namespace(
                    dataset=dataset,
                    data_dir=(
                        answers["data_dir"]
                        if answers["data_dir"] != default_data_dir
                        else None
                    ),
                    download=answers["download"],
                    output_dir=(
                        answers["output_dir"]
                        if answers["output_dir"] != default_preprocessed_dir
                        else None
                    ),
                    parse_entities=False,  # Default value
                )

                # Special handling for REBEL dataset
                if dataset == "REBEL (Requires approx 90GB Ram)":
                    rebel_action_question = [
                        inquirer.List(
                            "rebel_action",
                            message="REBEL dataset selected. Choose an action:",
                            choices=[
                                "Redownload entities and relations",
                                "Use pre-existing entities and relations",
                            ],
                        ),
                    ]
                    rebel_action_answer = inquirer.prompt(rebel_action_question)

                    if (
                        rebel_action_answer["rebel_action"]
                        == "Use pre-existing entities and relations"
                    ):
                        # Update the `parse_entities` attribute
                        args.parse_entities = True

                script_path = SCRIPTS[args.dataset]
                use_file_no = args.dataset != "ConLL04"

                try:
                    load_and_run_module(script_path, args, use_file_no)
                    print(
                        Fore.GREEN
                        + f"Finished processing {dataset} and saved to JSON object(s)."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        elif action_answer["action"] == "2. Combine":
            # Ask for data directory
            data_dir_question = [
                inquirer.Text(
                    "data_dir",
                    message="Location of the data files. Leave blank for default",
                    default=default_data_dir,
                ),
            ]
            data_dir_answer = inquirer.prompt(data_dir_question)
            data_dir = (
                data_dir_answer["data_dir"]
                if data_dir_answer["data_dir"]
                else default_data_dir
            )

            # Get available datasets
            available_datasets = get_available_datasets(data_dir, default_data_dir)

            if not available_datasets:
                print(
                    Fore.RED
                    + "No datasets found. Returning to main menu."
                    + Style.RESET_ALL
                )
                continue

            # Ask for dataset selection if combining
            combine_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to combine",
                    choices=available_datasets,
                ),
            ]
            combine_answer = inquirer.prompt(combine_question)

            # Ask if user wants to combine into a single dataset or preprocess individually
            combine_or_preprocess_question = [
                inquirer.List(
                    "combine_or_preprocess",
                    message="Do you want to combine into a single dataset or finalize preprocessing individually?",
                    choices=[
                        "Combine into single dataset",
                        "Finalize preprocessing individually",
                    ],
                ),
            ]
            combine_or_preprocess_answer = inquirer.prompt(
                combine_or_preprocess_question
            )

            if (
                combine_or_preprocess_answer["combine_or_preprocess"]
                == "Combine into single dataset"
            ):
                # Create an argparse.Namespace object to hold the arguments for combining
                args = argparse.Namespace(
                    folders=[
                        str(Path(data_dir, f"{dataset}_Modified"))
                        for dataset in combine_answer["datasets"]
                    ],
                    output_dir=default_preprocessed_dir,
                    verbose=True,
                )

                try:
                    # Load and run the combine script
                    combine_script_path = "components/scripts/combining.py"
                    load_and_run_module(
                        combine_script_path, args, use_file_no=False, combine=True
                    )
                    print(
                        Fore.GREEN
                        + "Finished combining datasets and saved to JSON object(s)."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"An error occurred while combining datasets: {e}"
                        + Style.RESET_ALL
                    )
            else:
                for dataset in combine_answer["datasets"]:
                    # Create an argparse.Namespace object to hold the arguments for individual preprocessing
                    args = argparse.Namespace(
                        folders=[str(Path(data_dir, f"{dataset}_Modified"))],
                        output_dir=str(Path(default_preprocessed_dir, dataset)),
                        verbose=True,
                    )

                    try:
                        # Load and run the combine script for each dataset individually
                        combine_script_path = "components/scripts/combining.py"
                        load_and_run_module(
                            combine_script_path, args, use_file_no=False, combine=True
                        )
                        print(
                            Fore.GREEN
                            + f"Finished preprocessing {dataset} and saved to JSON object(s)."
                            + Style.RESET_ALL
                        )
                    except Exception as e:
                        print(
                            Fore.RED
                            + f"An error occurred while preprocessing {dataset}: {e}"
                            + Style.RESET_ALL
                        )
        elif action_answer["action"] == "3. Export JSON types":
            # Ask for preprocessed data directory
            preprocessed_dir_question = [
                inquirer.Text(
                    "preprocessed_dir",
                    message="Location of the pre-processed data files. Leave blank for default",
                    default=default_preprocessed_dir,
                ),
            ]
            preprocessed_dir_answer = inquirer.prompt(preprocessed_dir_question)
            preprocessed_dir = (
                preprocessed_dir_answer["preprocessed_dir"]
                if preprocessed_dir_answer["preprocessed_dir"]
                else default_preprocessed_dir
            )

            # Get available datasets in preprocessed directory
            available_datasets = scan_preprocessed_data(preprocessed_dir)

            if not available_datasets:
                print(
                    Fore.RED
                    + "No preprocessed datasets found. Returning to main menu."
                    + Style.RESET_ALL
                )
                continue

            # Ask for dataset selection for exporting JSON types
            export_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to export JSON types",
                    choices=available_datasets,
                ),
            ]
            export_answer = inquirer.prompt(export_question)

            for dataset in export_answer["datasets"]:
                input_file = Path(preprocessed_dir, dataset)
                output_dir = Path(preprocessed_dir, dataset)

                # Create an argparse.Namespace object to hold the arguments for exporting JSON types
                args = argparse.Namespace(
                    input_file=str(input_file),
                    output_dir=str(output_dir),
                )

                try:
                    # Load and run the Excel2Json script
                    excel2json_script_path = "components/scripts/Excel2Json.py"
                    load_and_run_module(
                        excel2json_script_path,
                        args,
                        use_file_no=False,
                        combine=False,
                        JSON=True,
                    )
                    print(
                        Fore.GREEN
                        + f"Finished exporting JSON types for {dataset}."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"An error occurred while exporting JSON types for {dataset}: {e}"
                        + Style.RESET_ALL
                    )
        elif action_answer["action"] == "4. Clean":
            # Ask for preprocessed data directory
            preprocessed_dir_question = [
                inquirer.Text(
                    "preprocessed_dir",
                    message="Location of the pre-processed data files. Leave blank for default",
                    default=default_preprocessed_dir,
                ),
            ]
            preprocessed_dir_answer = inquirer.prompt(preprocessed_dir_question)
            preprocessed_dir = (
                preprocessed_dir_answer["preprocessed_dir"]
                if preprocessed_dir_answer["preprocessed_dir"]
                else default_preprocessed_dir
            )

            # Get available datasets in preprocessed directory
            available_datasets = scan_preprocessed_data(preprocessed_dir)

            if not available_datasets:
                print(
                    Fore.RED
                    + "No preprocessed datasets found. Returning to main menu."
                    + Style.RESET_ALL
                )
                continue

            # Ask for dataset selection for cleaning (updating types)
            clean_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to clean (update types)",
                    choices=available_datasets,
                ),
            ]
            clean_answer = inquirer.prompt(clean_question)

            for dataset in clean_answer["datasets"]:
                input_file = Path(preprocessed_dir, dataset)
                output_dir = Path(preprocessed_dir, dataset)

                # Create an argparse.Namespace object to hold the arguments for cleaning (updating types)
                args = argparse.Namespace(
                    input_file=str(input_file),
                    output_dir=str(output_dir),
                )

                try:
                    # Load and run the cleaner script
                    cleaner_script_path = "components/scripts/cleaner.py"
                    load_and_run_module(
                        cleaner_script_path,
                        args,
                        use_file_no=False,
                        combine=False,
                        JSON=True,
                    )
                    print(
                        Fore.GREEN
                        + f"Finished cleaning (updating types) for {dataset}."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"An error occurred while cleaning (updating types) for {dataset}: {e}"
                        + Style.RESET_ALL
                    )
        elif action_answer["action"] == "5. Verify":
            # Ask for preprocessed data directory
            preprocessed_dir_question = [
                inquirer.Text(
                    "preprocessed_dir",
                    message="Location of the pre-processed data files. Leave blank for default",
                    default=default_preprocessed_dir,
                ),
            ]
            preprocessed_dir_answer = inquirer.prompt(preprocessed_dir_question)
            preprocessed_dir = (
                preprocessed_dir_answer["preprocessed_dir"]
                if preprocessed_dir_answer["preprocessed_dir"]
                else default_preprocessed_dir
            )

            # Get available datasets in preprocessed directory
            available_datasets = scan_preprocessed_data(preprocessed_dir)

            if not available_datasets:
                print(
                    Fore.RED
                    + "No preprocessed datasets found. Returning to main menu."
                    + Style.RESET_ALL
                )
                continue

            # Ask for dataset selection for verification
            verify_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to verify",
                    choices=available_datasets,
                ),
            ]
            verify_answer = inquirer.prompt(verify_question)

            # Ask for model selection
            model_question = [
                inquirer.List(
                    "model",
                    message="Select the model to use for verification",
                    choices=["bert-base-cased", "roberta-large"],
                ),
            ]
            model_answer = inquirer.prompt(model_question)

            # Ask for max sequence length
            max_seq_length_question = [
                inquirer.Text(
                    "max_seq_length",
                    message="Specify the maximum sequence length (default is 1024)",
                    default="1024",
                ),
            ]
            max_seq_length_answer = inquirer.prompt(max_seq_length_question)
            max_seq_length = int(max_seq_length_answer["max_seq_length"])

            for dataset in verify_answer["datasets"]:
                if dataset == "ALL (COMBINED)":
                    input_file = Path(
                        preprocessed_dir, "combined_dataset_corrected.json"
                    )
                else:
                    input_file = Path(
                        preprocessed_dir, dataset, "combined_dataset_corrected.json"
                    )

                output_dir = (
                    preprocessed_dir
                    if dataset == "ALL (COMBINED)"
                    else Path(preprocessed_dir, dataset)
                )

                # Create an argparse.Namespace object to hold the arguments for verification
                args = argparse.Namespace(
                    input_file=str(input_file),
                    output_dir=str(output_dir),
                    model_name=model_answer["model"],
                    max_seq_length=max_seq_length,
                )

                try:
                    # Load and run the verify script
                    verify_script_path = "components/scripts/verify.py"
                    load_and_run_module(
                        verify_script_path,
                        args,
                        use_file_no=False,
                        combine=False,
                        JSON=False,
                        verify=True,
                    )
                    print(
                        Fore.GREEN
                        + f"Finished verification for {dataset}."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"An error occurred while verifying {dataset}: {e}"
                        + Style.RESET_ALL
                    )
        elif action_answer["action"] == "6. Split":
            # Ask for preprocessed data directory
            preprocessed_dir_question = [
                inquirer.Text(
                    "preprocessed_dir",
                    message="Location of the pre-processed data files. Leave blank for default",
                    default=default_preprocessed_dir,
                ),
            ]
            preprocessed_dir_answer = inquirer.prompt(preprocessed_dir_question)
            preprocessed_dir = (
                preprocessed_dir_answer["preprocessed_dir"]
                if preprocessed_dir_answer["preprocessed_dir"]
                else default_preprocessed_dir
            )

            # Get available datasets containing corrected JSON files
            available_datasets = scan_corrected_files(preprocessed_dir)

            if not available_datasets:
                print(
                    Fore.RED
                    + "No datasets with corrected JSON files found. Returning to main menu."
                    + Style.RESET_ALL
                )
                continue

            # Ask for dataset selection for splitting
            split_question = [
                inquirer.Checkbox(
                    "datasets",
                    message="Select the datasets to split",
                    choices=available_datasets,
                ),
            ]
            split_answer = inquirer.prompt(split_question)

            # Ask for output directory
            output_dir_question = [
                inquirer.Text(
                    "output_dir",
                    message="Location to save the split data files. Leave blank for default",
                    default=default_finalized_dir,
                ),
            ]
            output_dir_answer = inquirer.prompt(output_dir_question)
            output_dir = (
                output_dir_answer["output_dir"]
                if output_dir_answer["output_dir"]
                else default_finalized_dir
            )

            # Ask if user wants to generate new distant data
            gen_new_distant_question = [
                inquirer.Confirm(
                    "gen_new_distant",
                    message="Generate new distant data split?",
                    default=True,
                ),
            ]
            gen_new_distant_answer = inquirer.prompt(gen_new_distant_question)

            for dataset in split_answer["datasets"]:
                if dataset == "ALL (COMBINED)":
                    input_file = Path(
                        preprocessed_dir, "combined_dataset_corrected.json"
                    )
                else:
                    input_file = Path(
                        preprocessed_dir, dataset, "combined_dataset_corrected.json"
                    )

                dataset_output_dir = (
                    output_dir
                    if dataset == "ALL (COMBINED)"
                    else Path(output_dir, dataset)
                )

                # Create an argparse.Namespace object to hold the arguments for splitting
                args = argparse.Namespace(
                    input_file=str(input_file),
                    output_dir=str(dataset_output_dir),
                    dataset_name=dataset,
                    gen_new_distant=gen_new_distant_answer["gen_new_distant"],
                )

                try:
                    # Load and run the split script
                    split_script_path = "components/scripts/split.py"
                    load_and_run_module(
                        split_script_path,
                        args,
                        use_file_no=False,
                        combine=False,
                        JSON=False,
                        verify=False,
                        split=True,
                    )

                    # Move the corresponding DREEAM_META folder
                    meta_folder = Path(preprocessed_dir, dataset, "DREEAM_META")
                    if meta_folder.exists():
                        final_meta_folder = Path(output_dir, dataset, "DREEAM_META")
                        final_meta_folder.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(meta_folder), str(final_meta_folder))
                        print(
                            Fore.GREEN
                            + f"Moved {meta_folder} to {final_meta_folder}."
                            + Style.RESET_ALL
                        )

                    # Move the corresponding REBEL_META folder
                    meta_folder = Path(preprocessed_dir, dataset, "REBEL_META")
                    if meta_folder.exists():
                        final_meta_folder = Path(
                            output_dir,
                            dataset,
                            "REBEL_META",
                        )
                        final_meta_folder.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(meta_folder), str(final_meta_folder))
                        print(
                            Fore.GREEN
                            + f"Moved {meta_folder} to {final_meta_folder}."
                            + Style.RESET_ALL
                        )

                    print(
                        Fore.GREEN
                        + f"Finished splitting and moving meta for {dataset}."
                        + Style.RESET_ALL
                    )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"An error occurred while splitting {dataset}: {e}"
                        + Style.RESET_ALL
                    )
        break


if __name__ == "__main__":
    init(autoreset=True)
    main()
