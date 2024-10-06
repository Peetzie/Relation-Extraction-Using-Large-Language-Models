# docred_distant_parser.py
# Created by: Frederik Peetz-Schou Larsen
# Description: Script to parse and process the DocRED dataset.
# This script loads JSON files, processes named entity recognition (NER) and relation extraction data,
# and outputs the processed data to JSON files.
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent))

import json
import pandas as pd
import glob
from pathlib import Path
import argparse
from colorama import Fore, Style, init
import gdown
import os
import shutil
from tqdm import tqdm

from download_utils import *


# def sentences_to_dict(sentences):
#     return [{"sent_id": i, "sent": sentence} for i, sentence in enumerate(sentences)]


def load_data(json_files, file_no, types_file):
    # Load the JSON file
    data_org = read_json_with_progress_no_lines(json_files[file_no])

    df_data = {
        "org_dataset": "DocRED_Distant",
        "title": [],
        "sents": [],
        "vertexSet": [],
        "labels": [],
    }
    print(json_files[file_no])

    for doc in tqdm(data_org, desc="Loading data", total=len(data_org)):
        df_data["title"].append(doc["title"])
        df_data["sents"].append([sent for sent in doc["sents"]])
        df_data["vertexSet"].append(doc["vertexSet"])
        if "labels" in doc:
            df_data["labels"].append(doc["labels"])
        else:
            df_data["labels"].append([])

    df = pd.DataFrame(df_data)

    with open(types_file) as f:
        types = json.load(f)

    relations_file = types

    return df, relations_file


# def df_generate_examples(df, relations_file):
#     examples_df = []
#     for id_, row in df.iterrows():
#         triplets = ""
#         prev_head = None
#         relations_sorted = sorted(row["relations"], key=lambda tup: tup["h"])
#         for relation in relations_sorted:
#             if prev_head == relation["h"]:
#                 triplets += (
#                     f' {mapping_types(row["NER"][relation["h"]][0]["type"])} '
#                     + row["NER"][relation["t"]][0]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]][0]["type"])} '
#                     + str(relations_file.get(relation["r"]))
#                 )
#             elif prev_head is None:
#                 triplets += (
#                     "<triplet> "
#                     + row["NER"][relation["h"]][0]["name"]
#                     + f' {mapping_types(row["NER"][relation["h"]][0]["type"])} '
#                     + row["NER"][relation["t"]][0]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]][0]["type"])} '
#                     + str(relations_file.get(relation["r"]))
#                 )
#                 prev_head = relation["h"]
#             else:
#                 triplets += (
#                     "<triplet> "
#                     + row["NER"][relation["h"]][0]["name"]
#                     + f' {mapping_types(row["NER"][relation["h"]][0]["type"])} '
#                     + row["NER"][relation["t"]][0]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]][0]["type"])} '
#                     + str(relations_file.get(relation["r"]))
#                 )
#                 prev_head = relation["h"]
#         examples_df.append(
#             {
#                 "title": row["title"],
#                 "sentences": " ".join(row["sentences"]),
#                 "id": row["title"],
#                 "triplets": triplets,
#             }
#         )
#     return examples_df


def download_and_manage_files(folder_id, dest_path):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, quiet=False, output=dest_path, use_cookies=False)

    files_to_keep = {
        "dev.json",
        "rel_info.json",
        "train_annotated.json",
        "train_distant.json",
        "test.json",
    }

    # Remove non-specified JSON files
    for root, dirs, files in os.walk(dest_path):
        for file in files:
            if file not in files_to_keep:
                os.remove(os.path.join(root, file))

    # remove the folder DocRED_baseline_metadata
    shutil.rmtree(os.path.join(dest_path, "DocRED_baseline_metadata"))


# def flatten_list_of_lists(list_of_lists):
#     """Flatten a list of lists into a single list."""
#     flat_list = [item for sublist in list_of_lists for item in sublist]
#     return flat_list


# def flattenassist(df):
#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Flattening NER"):
#         df.at[i, "NER"] = flatten_list_of_lists(row["NER"])
#     return df


def fix_types(df, relations_file):
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


def main(data_dir=None, output_dir=None, download=True, verbose=True):
    if data_dir is None:
        data_dir = Path(root_folder, "data", "raw_data")
    docred_data_dir = Path(data_dir, "DocRED_Distant")
    if docred_data_dir.exists() is False:
        docred_data_dir = Path(data_dir, "DocRED_Distant")
    if output_dir is None:
        output_dir = Path(data_dir, "DocRED_Distant_Modified")
        # if it does not exist create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    output_dir = Path(output_dir)
    data_dir = str(docred_data_dir / "*.json")

    if download:
        folder_id = "1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw"
        download_and_manage_files(folder_id, str(docred_data_dir))

    json_files = glob.glob(data_dir)
    types_file = [file for file in json_files if "rel_info.json" in file][0]
    json_files = [file for file in json_files if "rel_info.json" not in file]
    # Copy over the relations file
    shutil.copy(types_file, output_dir)
    # if file_no == -1:
    dfs = []
    print(Fore.RED + "Processing all files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(json_files)):
        print(Fore.GREEN + f"Processing file {i+1}/{len(json_files)}" + Fore.WHITE)
        df, _ = load_data(json_files, i, types_file)
        # df["sentences"] = df["sentences"].apply(sentences_to_dict)

        df["file_path"] = json_files[i]
        df["domains"] = "general"
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = fix_types(df, types_file)
    # df = flattenassist(df)

    df = save_json_with_progress(df, Path(output_dir, "DocRED_Distant_modified.json"))
    print(
        f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'DocRED_Distant_modified.json')}{Style.RESET_ALL}"
    )
    print(df.head())
    print(df.columns)

    # # else:
    #     print(Fore.GREEN + f"Processing file {file_no}" + Fore.WHITE)
    #     df, _ = load_data(json_files, file_no, types_file)
    #     df["sentences"] = df["sentences"].apply(sentences_to_dict)

    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     # add the domains type
    #     df["domains"] = "general"
    #     df = save_json_with_progress(
    #         df, Path(output_dir, "DocRED_Distant_modified.json")
    #     )
    #     print(
    #         f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'DocRED_Distant_modified.json')}{Style.RESET_ALL}"
    #     )
    #     print(df.head())
    #     print(df.columns)

    #     # examples = generate_examples_test(
    #     #     Path(output_dir, f"{filename}.json"), lines=False
    #     # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse DocRED (Distant) dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="Location to match JSON files.",
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
        help="Download the dataset from Google Drive.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.output_dir,
        args.download,
        args.verbose,
    )
    print(Fore.GREEN + "Finished processing file(s) and saved to JSON object(s)")
