# conll04_prepare.py
# Created by: Frederik Peetz-Schou Larsen
# Description: Script to download and process the CoNLL04 dataset.
# This script loads JSON files, processes named entity recognition (NER) and relation extraction data,
# and outputs the processed data to JSON files.

import sys
from pathlib import Path
import argparse

from colorama import Style

sys.path.append(str(Path(__file__).resolve().parent))

import json
import pandas as pd
import glob
import os
from tqdm import tqdm
import requests

from download_utils import *


def load_json_files(data_dir, files):
    data = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        json_data = read_json_with_progress_no_lines(os.path.join(data_dir, file))
        for item in json_data:
            item["original_file_path"] = filepath
        data.extend(json_data)
    return data


def transform_data_to_custom_format(data, file_path):
    result = []
    for item in tqdm(data, desc="Transforming data to custom format", total=len(data)):
        tokens = item["tokens"]
        entities = item["entities"]
        relations = item["relations"]
        sent_id = item.get("orig_id", len(result))
        fpath = item["original_file_path"]

        vertex_set = []
        for entity in entities:
            entity_type = entity["type"]
            start = entity["start"]
            end = entity["end"]
            vertex_set.append(
                [
                    {
                        "name": " ".join(tokens[start:end]),
                        "pos": [start, end],
                        "type": entity_type,
                        "sent_id": 0,
                    }
                ]
            )

        labels = []
        for relation in relations:
            labels.append(
                {
                    "r": relation["type"],
                    "h": relation["head"],
                    "t": relation["tail"],
                    "evidence": [0],
                }
            )

        result.append(
            {
                "org_dataset": "CoNLL04",
                "title": str(sent_id),
                "sents": [tokens],
                "vertexSet": vertex_set,
                "labels": labels,
                "file_path": str(fpath),
                "domains": "general",
            }
        )
    return result


def main(data_dir=None, output_dir=None, download=True, verbose=True):
    if data_dir is None:
        data_dir = Path(root_folder, "data", "raw_data")
    conll04_data_dir = Path(data_dir, "CoNLL04")
    if not os.path.exists(conll04_data_dir):
        os.makedirs(conll04_data_dir)

    if output_dir is None:
        output_dir = Path(data_dir, "CoNLL04_Modified")

    if download:
        print("Downloading the data.")
        base_url = "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04"
        files = [
            "conll04_train.json",
            "conll04_dev.json",
            "conll04_test.json",
            # "conll04_train_dev.json",# Not included in the original REBEL SPLIT. Therefore removed here.
        ]
        download_files_Conll04(base_url, conll04_data_dir, files)

    files_to_load = [
        "conll04_train.json",
        "conll04_dev.json",
        "conll04_test.json",
        # "conll04_train_dev.json", # Not included in the original REBEL SPLIT. Therefore removed here.
    ]
    data = load_json_files(conll04_data_dir, files_to_load)
    transformed_data = transform_data_to_custom_format(data, conll04_data_dir)
    df = pd.DataFrame(transformed_data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "Conll04_processed.json")
    save_json_with_progress(df, output_file)
    print(f"{Fore.GREEN}Dataframe saved to {output_file}.{Style.RESET_ALL}")
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process CoNLL04 dataset."
    )
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
        help="Download the dataset from the specified URL.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.output_dir,
        args.download,
        args.verbose,
    )
    print("Finished processing file(s) and saved to JSON object(s).")
