import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent))
import json
import pandas as pd
import glob
from colorama import Fore, Style, init
import os
import shutil
from tqdm import tqdm
from download_utils import *


def load_data(json_files, file_no, types_file=None):
    # Load the JSON file
    with open(json_files[file_no]) as f:
        data_org = json.load(f)

    df_data = {
        "org_dataset": "ReDocRED",
        "title": [],
        "sents": [],
        "vertexSet": [],
        "labels": [],
    }
    print(json_files[file_no])

    for doc in data_org:
        df_data["title"].append(doc["title"])
        df_data["sents"].append([sent for sent in doc["sents"]])
        df_data["vertexSet"].append(doc["vertexSet"])
        if "labels" in doc:
            df_data["labels"].append(doc["labels"])
        else:
            df_data["labels"].append([])

    df = pd.DataFrame(df_data)

    if types_file:
        with open(types_file) as f:
            types = json.load(f)

        relations_file = types
        return df, relations_file
    else:
        return df


def fix_types(df, relations_file=None):
    j_file = open(relations_file)
    types = json.load(j_file)
    relations_updated = 0
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Preprocessing initial relations"
    ):
        relations = row["labels"]
        for rel in relations:
            for _type in types.keys():
                if _type == rel["r"]:
                    relations_updated += 1
                    rel["r"] = types[_type]
    print(f"Updated {relations_updated} relations")
    return df


def main(data_dir=None, file_no=-1, output_dir=None, download=True, verbose=True):
    if data_dir is None:

        data_dir = Path(root_folder, "data", "raw_data")
    redocred_data_dir = Path(data_dir, "ReDocRED")
    if redocred_data_dir.exists() is False:
        os.makedirs(redocred_data_dir)
    if output_dir is None:
        output_dir = Path(data_dir, "ReDocRED_Modified")
        # if it does not exist create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    output_dir = Path(output_dir)
    data_dir = str(redocred_data_dir / "*.json")

    if download:
        base_url = "https://github.com/tonytan48/Re-DocRED/tree/main/data"
        download_json_files_from_github(
            repo_url=base_url, download_folder=redocred_data_dir
        )

    json_files = glob.glob(data_dir)

    if file_no == -1:
        dfs = []
        print(Fore.RED + "Processing all files.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(json_files)):
            print(Fore.GREEN + f"Processing file {i+1}/{len(json_files)}" + Fore.WHITE)
            df = load_data(json_files, i)
            # df["sentences"] = df["sentences"].apply(sentences_to_dict)

            df["file_path"] = json_files[i]
            df["domains"] = "general"
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        types_file_path = "/work3/s174159/LLM_Thesis/databuilding/types/rel_info.json"
        df = fix_types(df, types_file_path)
        # df = flattenassist(df)

        df = save_json_with_progress(
            df, Path(output_dir, "ReDocRED_Joint_modified.json")
        )
        print(
            f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'ReDocRED_Joint_modified.json')}{Style.RESET_ALL}"
        )
        print(df.head())
        print(df.columns)

    else:
        print(Fore.GREEN + f"Processing file {file_no}" + Fore.WHITE)
        df = load_data(json_files, file_no)
        df["sentences"] = df["sentences"].apply(sentences_to_dict)

        output_dir.mkdir(parents=True, exist_ok=True)
        df["domains"] = "general"
        df = save_json_with_progress(
            df, Path(output_dir, "ReDocRED_Distant_modified.json")
        )
        print(
            f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'ReDocRED_Distant_modified.json')}{Style.RESET_ALL}"
        )
        print(df.head())
        print(df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ReDocRED (Distant) dataset.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="Location to match JSON files.",
    )
    parser.add_argument(
        "--file_no",
        type=int,
        required=True,
        help="File number to process. -1 for all files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory to save results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Prints out the first example to the console.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from the provided URL.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.file_no,
        args.output_dir,
        args.download,
        args.verbose,
    )
    print(Fore.GREEN + "Finished processing file(s) and saved to JSON object(s)")
