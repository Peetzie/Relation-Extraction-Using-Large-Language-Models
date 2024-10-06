import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent))

import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import argparse
from colorama import Fore, Style, init

from download_utils import *

REPO_URL = "https://github.com/mainlp/CrossRE/tree/main/crossre_data"
GOLD_STD_URL = "https://github.com/mainlp/CrossRE/tree/main/crossre_annotation/last_annotation_round"


def load_json_files(json_files):
    dataframes = []
    print(json_files)
    for file in tqdm(json_files):
        ## Add a column in the dataframe with the name of the dataset (cross-re)
        temp_df = pd.read_json(file, orient="records", lines=True)
        temp_df["org_dataset"] = "CrossRE"
        temp_df["domains"] = file.split("/")[-1].split(".")[0].split("-")[0]
        temp_df["file_path"] = file
        # make it the first column
        cols = temp_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        temp_df = temp_df[cols]
        temp_df = temp_df.rename(columns={"sentences": "sents"})

        dataframes.append(temp_df)
    df = pd.concat(dataframes, ignore_index=True)
    return df


def transform_words_to_sentence(word_list):
    sent_list = []
    """
    Transforms a list of words into a coherent sentence.
    
    Args:
    word_list (list): List of words.
    
    Returns:
    str: A single sentence formed by concatenating the words.
    """
    # Join the words into a single string
    sentence = " ".join(word_list)
    sent_dict = {"sent_id": 0, "sent": sentence}
    sent_list.append(sent_dict)
    return sent_list


def transform_sentences(df):
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Transforming sentences to dicts"
    ):
        # apply the function to the sentence column
        df.at[i, "sents"] = transform_words_to_sentence(row["sentence"])
    return df


def process_ners(df):
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing NERS"):

        # For each NER in the list of NERS transform it to a list of list with dictionaries in this format: [{'pos': [1, 6], 'type': 'ORG', 'sent_id': 0, 'name': 'Worker-Peasant Red Guards'}] where pos is the start and end, type is the 3rd item in current list. And name should be fetched from the sentence
        ner_list = []
        for ner in row["ner"]:
            pos = [int(ner[0]), int(ner[1])]
            name = []
            word_range = range(pos[0], pos[1] + 1)
            for word in word_range:
                name.append(df.iloc[i]["sents"][0].get("sent").split(" ")[word])
            ner_dict = {
                "name": " ".join(name),
                "pos": [int(ner[0]), int(ner[1]) + 1],
                "type": ner[2],
                "sent_id": 0,
            }
            ner_list.append(ner_dict)
        df.at[i, "ner"] = ner_list
    return df


def process_relations(df):
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing relations"):
        relations_list = []

        for relation in row["relations"]:
            rel1_start = relation[0]
            rel1_end = relation[1]
            rel2_start = relation[2]
            rel2_end = relation[3]
            rel_type = relation[4]

            updated_relation = [
                relation[0],
                relation[1],
                relation[2],
                relation[3],
                relation[4],
            ]

            for j, ner in enumerate(row["ner"]):
                ner_start = ner["pos"][0]
                ner_end = ner["pos"][1]

                if ner_start == rel1_start and ner_end - 1 == rel1_end:
                    updated_relation[0] = j
                if ner_start == rel2_start and ner_end - 1 == rel2_end:
                    updated_relation[2] = j

            # Verify indices are within the range of the ner list
            if 0 <= updated_relation[0] < len(row["ner"]) and 0 <= updated_relation[
                2
            ] < len(row["ner"]):
                tmp_rel = {
                    "r": rel_type,
                    "h": updated_relation[0],
                    "t": updated_relation[2],
                    "evidence": [0],  # sentence_id => 0 => there is only one sentence
                }
                relations_list.append(tmp_rel)
            else:
                print(f"Invalid index found in row {i}: {updated_relation}")

        df.at[i, "relations"] = relations_list
    return df


def change_cols(df):
    # change columns of dataframe  to org_dataset	title	sentences	NER	relations

    df = df[
        ["org_dataset", "doc_key", "domains", "sents", "ner", "relations", "file_path"]
    ]
    df = df.rename(
        columns={"doc_key": "title", "relations": "labels", "ner": "vertexSet"}
    )
    return df


def check_integrity(df, dfc):
    # check that the length of relations are identical to before and after the transformation
    no_of_deviations = 0
    for i, row in df.iterrows():
        relations_list_length = len(row["labels"])
        if len(dfc.iloc[i]["labels"]) != relations_list_length:
            no_of_deviations += 1
            print(
                f"Row {i} has a different number of relations in the modified dataframe. Before: {len(dfc.iloc[i]['relations'])}, After: {relations_list_length}"
            )

    if no_of_deviations == 0:
        print(
            "All relations are identical in length before and after the transformation."
        )


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


def remove_empty_vertexSet(df):
    df = df[df["vertexSet"].apply(len) > 0]
    return df


def filter_dataframe(df):
    """
    Filters the dataframe to find rows where vertexSet only has one entity
    which is only mentioned once and the labels list is empty.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    filtered_df = df[
        ~(
            (df["vertexSet"].apply(len) == 1)
            & (df["vertexSet"].apply(lambda x: len(x[0]) == 1))
            & (df["labels"].apply(len) == 0)
        )
    ]

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def main(data_dir=None, file_no=-1, output_dir=None, download=True, verbose=True):
    if data_dir is None:
        data_dir = Path(root_folder, "data", "raw_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    crossRE_data_dir = str(Path(data_dir, "CrossRE"))
    # if folder doesnt exist create it
    if not os.path.exists(crossRE_data_dir):
        os.makedirs(crossRE_data_dir)

    if download:
        download_json_files_from_github(REPO_URL, crossRE_data_dir, verbose)
        download_json_files_from_github(GOLD_STD_URL, crossRE_data_dir, verbose)

    # Load the JSON files
    json_files = list_files_in_folder(crossRE_data_dir)

    if file_no == -1:
        # Do all files else single file
        json_files = json_files
    else:
        json_files = json_files[file_no]

    df = load_json_files(json_files)
    df = transform_sentences(df)
    print(df)
    df = process_ners(df)
    print(df)
    df = process_relations(df)
    df = change_cols(df)
    dfc = df.copy()
    check_integrity(df, dfc)
    df = fix_grouped_entities(df)
    df = back_to_words(df)
    df = remove_empty_vertexSet(df)
    df = filter_dataframe(df)

    if output_dir is None:
        output_dir = Path(data_dir, "CrossRE_Modified")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.path.join(output_dir, "CrossRE_modified.json")
    output_file = os.path.join(output_dir, "CrossRE_modified.json")
    save_json_with_progress(df, output_file)
    print(f"{Fore.GREEN}Dataframe saved to {output_file}.{Style.RESET_ALL}")
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process JSON files from a GitHub repository."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="The directory to download JSON files to or load from.",
    )
    parser.add_argument(
        "--file_no", type=int, default=None, help="Number of JSON files to load."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The directory to save the processed data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Flag to download JSON files from GitHub.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity."
    )

    args = parser.parse_args()
    main(
        args.data_dir,
        args.file_no,
        args.output_dir,
        args.download,
        args.verbose,
    )
