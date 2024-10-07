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
import os
import requests
import tarfile
from tqdm import tqdm


from download_utils import *


def download_and_extract(url, download_folder="download", verbose=False):
    if verbose:
        print(f"{Fore.GREEN}Downloading data from {url}...{Style.RESET_ALL}")

    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    # Define the path for the downloaded file
    tar_gz_path = os.path.join(download_folder, "sciERC_processed.tar.gz")

    # Download the file with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(tar_gz_path, "wb") as file, tqdm(
        desc=tar_gz_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            file.write(chunk)
            bar.update(len(chunk))

    if verbose:
        print(f"{Fore.GREEN}Extracting data to {download_folder}...{Style.RESET_ALL}")

    # Extract the tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:

        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tar, path=download_folder)

    if verbose:
        print(f"{Fore.GREEN}Cleaning up non-JSON files and folders...{Style.RESET_ALL}")

    # Move JSON files out of any nested folders and delete those folders
    json_files = []
    for root, dirs, files in os.walk(download_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    new_file_path = os.path.join(download_folder, file)
                    os.rename(file_path, new_file_path)
                    json_files.append(new_file_path)

    # Remove the processed_data and all subfolders
    for root, dirs, files in os.walk(download_folder, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for nested_root, nested_dirs, nested_files in os.walk(
                dir_path, topdown=False
            ):
                for nested_file in nested_files:
                    os.remove(os.path.join(nested_root, nested_file))
                for nested_dir in nested_dirs:
                    os.rmdir(os.path.join(nested_root, nested_dir))
            os.rmdir(dir_path)

    # Remove the tar.gz file after extraction
    os.remove(tar_gz_path)

    if verbose:
        print(f"{Fore.GREEN}Download and extraction completed.{Style.RESET_ALL}")

    return json_files


def map_positions_to_entity_indices(relations, entities):
    """Map relation positions to entity indices and create a list of relation dictionaries."""
    pos_to_entity_idx = {}
    for idx, entity in enumerate(entities):
        start_pos, end_pos = entity["pos"]
        sent_id = int(entity["sent_id"])
        for pos in range(start_pos, end_pos + 1):
            pos_to_entity_idx[pos] = (idx, sent_id)

    new_relations = []
    for relation in relations:
        new_relation = []
        for rel in relation:
            head_start, head_end, tail_start, tail_end, rel_type = rel
            head_info = pos_to_entity_idx.get(head_start, (-1, -1))
            tail_info = pos_to_entity_idx.get(tail_start, (-1, -1))
            head_idx, head_sent_id = head_info
            tail_idx, tail_sent_id = tail_info

            if head_idx != -1 and tail_idx != -1:
                new_relation.append(
                    {
                        "r": rel_type,
                        "h": head_idx,
                        "t": tail_idx,
                        "evidence": list(set([head_sent_id, tail_sent_id])),
                    }
                )
        new_relations.append(new_relation)
    return new_relations


def adjust_entity_word_idx_relative_to_sentence_position(df):
    for i, row in df.iterrows():
        previous_sent_length = {}
        total_length = 0
        for sent in row["sents"]:
            previous_sent_length[int(sent["sent_id"])] = total_length
            total_length += len(sent["sent"].split())
        for ent in row["vertexSet"]:
            sent_id = ent["sent_id"]
            if sent_id == 0:
                continue
            start, end = ent["pos"]
            offset = previous_sent_length[sent_id]
            ent["pos"] = [start - offset, end - offset]
    return df


def main(data_dir=None, file_no=-1, output_dir=None, download=True, verbose=True):
    if data_dir is None:

        data_dir = Path(root_folder, "data", "raw_data")
    sciERC_data_dir = Path(data_dir, "SciERC")

    if output_dir is None:
        output_dir = Path(data_dir, "SciERC_Modified")

    # Download the dataset if download argument is provided
    if download:
        url = "https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
        download_and_extract(url, sciERC_data_dir, verbose)  # type: ignore

    json_files = glob.glob(str(sciERC_data_dir / "*.json"))
    if file_no == -1:
        print(Fore.RED + "Processing all files.")
        dfs = []
        for i in range(len(json_files)):
            print(i)
            print(Fore.GREEN + f"Processing file {i+1}/{len(json_files)}")
            df_sciERC = process_file(json_files, i)
            dfs.append(df_sciERC)
        print(len(dfs))
        df = pd.concat(dfs, ignore_index=True)
        df = adjust_entity_word_idx_relative_to_sentence_position(df)
        df = back_to_words(df)
        df = fix_grouped_entities(df)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = save_json_with_progress(df, Path(output_dir, "SciERC_modified.json"))
        print(
            f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'SciERC_modified.json')}{Style.RESET_ALL}"
        )
        print(df.head())
        print(df.columns)

    else:
        print(Fore.GREEN + f"Processing file {file_no}")
        df_sciERC = process_file(json_files, file_no)
        df = adjust_entity_word_idx_relative_to_sentence_position(df_sciERC)
        df = back_to_words(df)
        df = fix_grouped_entities(df)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = save_json_with_progress(df, Path(output_dir, "SciERC_modified.json"))
        print(
            f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'SciERC_modified.json')}{Style.RESET_ALL}"
        )
        print(df.head())
        print(df.columns)


def back_to_words(df):
    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Transforming sentences back to correct format",
    ):
        sentencesD = row["sents"]
        sent_list = []
        for sentence in sentencesD:
            sent_list.append(sentence["sent"].split())
        df.at[i, "sents"] = sent_list
    return df


def process_file(json_files, file_no):
    df_data = {
        "org_dataset": "SciERC",
        "title": [],
        "sentences": [],
        "NER": [],
        "relations": [],
        "file_path": None,
    }

    filepath = json_files[file_no]
    sciERC_data = load_json_lines(filepath)

    for text in sciERC_data:
        df_data["title"].append(text["doc_key"])
        df_data["sentences"].append([" ".join(sent) for sent in text["sentences"]])
        df_data["NER"].append(text["ner"])
        df_data["relations"].append(text["relations"])
        df_data["file_path"] = filepath

    df_sciERC = pd.DataFrame(df_data)
    df_sciERC["sentences_elongated"] = df_sciERC["sentences"].apply(
        concatenate_sentences
    )
    df_sciERC["sentences_combined"] = df_sciERC["sentences"].apply(sentences_to_dict)
    df_sciERC["NER_changed"] = df_sciERC.apply(
        lambda row: dict_NER(row["NER"], row["sentences_elongated"]), axis=1
    )
    df_sciERC["relations_changed"] = df_sciERC.apply(
        lambda row: map_positions_to_entity_indices(
            row["relations"], row["NER_changed"]
        ),
        axis=1,
    )

    df_sciERC["domains"] = "science"

    df_sciERC = df_sciERC[
        [
            "org_dataset",
            "title",
            "domains",
            "sentences_combined",
            "NER_changed",
            "relations_changed",
            "file_path",
        ]
    ]
    df_sciERC = df_sciERC.rename(
        columns={
            "NER_changed": "vertexSet",
            "relations_changed": "labels",
            "sentences_combined": "sents",
        }
    )
    df_sciERC["labels"] = df_sciERC["labels"].apply(combine_list_of_lists)

    return df_sciERC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse SciERC dataset.")

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
        "--download",
        action="store_true",
        help="Download and extract the sciERC dataset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed log messages.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.file_no,
        args.output_dir,
        args.download,
        args.verbose,
    )
