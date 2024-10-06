import sys
import os
from pathlib import Path
import pandas as pd
import json
import argparse
from colorama import Fore, Style, init
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from download_utils import *


def get_data_split(df, dataset_name=None, output_dir=None, gen_new_distant=True):
    # Initial check for 'dataset' column
    if "dataset" not in df.columns and dataset_name is None:
        raise ValueError(
            "DataFrame must contain a 'dataset' column or a 'dataset_name' must be specified."
        )

    distant = None

    if dataset_name:  # Process specific dataset
        if dataset_name == "DocRED_Distant":
            print("Processing DocRED (Distant) data...")
            distant = df[df["original_file_path"].str.contains("distant")]
            un_distant = df[~df["original_file_path"].str.contains("distant")]
            test = un_distant[un_distant["original_file_path"].str.contains("test")]
            dev = un_distant[un_distant["original_file_path"].str.contains("dev")]
            train = un_distant[
                un_distant["original_file_path"].str.contains("train_annotated")
            ]
            print("size of datasets splits (distant)", len(distant))
            print("size of datasets splits (test)", len(test))
            print("size of datasets splits (train)", len(train))
            print("size of datasets splits (dev)", len(dev))
        elif dataset_name == "DocRED_Joint":
            test = df[df["original_file_path"].str.contains("test")]
            dev = df[df["original_file_path"].str.contains("dev")]
            train = df[df["original_file_path"].str.contains("train")]
        elif dataset_name == "ReDocRED":
            test = df[df["original_file_path"].str.contains("test")]
            dev = df[df["original_file_path"].str.contains("dev")]
            train = df[df["original_file_path"].str.contains("train")]
        elif dataset_name == "NYT":
            # if gen_new_distant:
            #     df, distant = train_test_split(df, train_size=0.1, random_state=42)
            #     dev, test = train_test_split(df, test_size=0.2, random_state=42)
            #     train, dev = train_test_split(dev, test_size=0.5, random_state=42)
            # else:
            test = df[df["original_file_path"].str.contains("test")]
            dev = df[df["original_file_path"].str.contains("valid")]
            train = df[df["original_file_path"].str.contains("train")]
        elif dataset_name == "SciERC":
            # if gen_new_distant:
            #     df, distant = train_test_split(df, train_size=0.1, random_state=42)
            #     dev, test = train_test_split(df, test_size=0.2, random_state=42)
            #     train, dev = train_test_split(dev, test_size=0.5, random_state=42)
            # else:
            test = df[df["original_file_path"].str.contains("test")]
            dev = df[df["original_file_path"].str.contains("dev")]
            train = df[df["original_file_path"].str.contains("train")]
        elif dataset_name == "CoNLL04":
            if gen_new_distant:
                df, distant = train_test_split(df, train_size=0.1, random_state=42)
                dev, test = train_test_split(df, test_size=0.2, random_state=42)
                train, dev = train_test_split(dev, test_size=0.5, random_state=42)
            else:
                dev = df[df["original_file_path"].str.contains("dev")]
                test = df[df["original_file_path"].str.contains("test")]
                train = df[df["original_file_path"].str.contains("train")]
        elif dataset_name == "CrossRE":
            if gen_new_distant:
                gold1 = df[df["original_file_path"].str.contains("NLP")]
                gold2 = df[df["original_file_path"].str.contains("linguist")]
                train = pd.concat([gold1, gold2])
                remainder = df[~df.index.isin(train.index)]
                remainder, distant = train_test_split(
                    remainder, test_size=0.8, random_state=42
                )
                dev, test = train_test_split(remainder, test_size=0.5, random_state=42)
            else:
                dev, test = train_test_split(df, test_size=0.2, random_state=42)
                train, dev = train_test_split(dev, test_size=0.5, random_state=42)
        elif dataset_name == "REBEL":
            if gen_new_distant:
                # Initialize test, dev, and train to avoid referencing before assignment
                test = pd.DataFrame()
                dev = pd.DataFrame()
                train = pd.DataFrame()
                distant = pd.concat([test, dev, train])
                remainder, distant = train_test_split(
                    distant, train_size=0.1, random_state=42
                )
                train, remainder = train_test_split(
                    df, train_size=0.66, random_state=42
                )
                test, dev = train_test_split(remainder, train_size=0.5, random_state=42)
            else:
                test = df[df["original_file_path"].str.contains("test")]
                dev = df[df["original_file_path"].str.contains("val")]
                train = df[df["original_file_path"].str.contains("train")]
        else:  # Combine all datasets
            print("Combining all datasets...")
            if gen_new_distant:
                distant_dfs, test_dfs, dev_dfs, train_dfs = [], [], [], []
                datasets = [
                    "DocRED_Distant",
                    "NYT",
                    "SciERC",
                    "CoNLL04",
                    "CrossRE",
                    "ReDocRED",
                ]
                for dataset in tqdm(
                    datasets, total=len(datasets), desc=f"Splitting datasets"
                ):
                    print("Current dataset:", dataset)
                    current_df = df[df["org_dataset"] == dataset]
                    train, dev, test, distant_split = get_data_split(
                        current_df,
                        dataset_name=dataset,
                        gen_new_distant=gen_new_distant,
                    )
                    if distant_split is not None:
                        distant_dfs.append(distant_split)
                    test_dfs.append(test)
                    dev_dfs.append(dev)
                    train_dfs.append(train)
                if distant_dfs:
                    distant = pd.concat(distant_dfs)
                test = pd.concat(test_dfs)
                dev = pd.concat(dev_dfs)
                train = pd.concat(train_dfs)
            else:
                test_dfs, dev_dfs, train_dfs = [], [], []
                datasets = [
                    "DocRED_Joint",
                    "NYT",
                    "SciERC",
                    "CoNLL04",
                    "CrossRE",
                    "ReDocRED",
                ]
                for dataset in tqdm(
                    datasets, total=len(datasets), desc=f"Splitting datasets"
                ):
                    current_df = df[df["org_dataset"] == dataset]
                    train, dev, test, _ = get_data_split(
                        current_df,
                        dataset_name=dataset,
                        gen_new_distant=gen_new_distant,
                    )
                    test_dfs.append(test)
                    dev_dfs.append(dev)
                    train_dfs.append(train)
                test = pd.concat(test_dfs)
                dev = pd.concat(dev_dfs)
                train = pd.concat(train_dfs)

    if output_dir is None:
        output_dir = str(
            Path("/work3/s174159/LLM_Thesis/data", "data_finalized", dataset_name)
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save files if output_dir is not None
    print(f"Saving files to {output_dir}")
    save_df_no_lines_json(train, Path(output_dir), "train_annotated.json")
    save_df_no_lines_json(dev, Path(output_dir), "dev.json")
    save_df_no_lines_json(test, Path(output_dir), "test.json")

    if gen_new_distant and distant is not None:
        save_df_in_multiple_files(
            distant, Path(output_dir), "train_distant.json", chunk_size=180000
        )

    return train, dev, test, distant


def main(input_file, output_dir, dataset_name, gen_new_distant):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_df = pd.DataFrame(read_json_with_progress_no_lines(input_file))

    train, dev, test, distant = get_data_split(
        combined_df, dataset_name, output_dir, gen_new_distant
    )
    print(f"Data split completed for {dataset_name}. Files saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train, dev, test, and distant sets"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The path to the combined dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the split dataset files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to process (e.g., DocRED, NYT, sciERC, CoNLL04, CrossRE, ALL)",
    )
    parser.add_argument(
        "--gen_new_distant",
        type=bool,
        default=True,
        help="Generate new distant data split",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.dataset_name, args.gen_new_distant)
