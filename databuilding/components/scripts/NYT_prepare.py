import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent))

import json
import zipfile
import pandas as pd
import glob
from pathlib import Path
import argparse
from colorama import Fore, Style, init
import gdown
import os
import shutil
import numpy as np
from tqdm import tqdm
import re
from download_utils import *

folder_id = "1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N"

# Initialize colorama
init(autoreset=True)


# Download and manage files function
def download_and_manage_files(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = os.path.join(dest_path, "downloaded_file.zip")

    os.makedirs(dest_path, exist_ok=True)
    gdown.download(url, zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    os.remove(zip_path)

    files_to_keep = {"raw_test.json", "raw_valid.json", "raw_train.json"}
    for root, dirs, files in os.walk(dest_path):
        for file in files:
            if file not in files_to_keep:
                os.remove(os.path.join(root, file))

    metadata_folder = os.path.join(folder_id, "NYT_baseline_metadata")
    if os.path.exists(metadata_folder):
        shutil.rmtree(metadata_folder)


# Concatenate sentences function
def concatenate_sentences(sentences):
    return "".join(sentences).split()


# Fix sentences function
def fix_sentences(df):
    sentences = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fixing sentences"):
        sent = row["sentText"]
        sentences.append(list(concatenate_sentences(sent)))

    df["sents"] = sentences
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Transforming sentences to lists"
    ):
        df.at[i, "sents"] = [row["sents"]]
    return df


# Find entity index function
def find_entity_index(vertex_set, entity_name):
    for index, entity in enumerate(vertex_set):
        if entity["name"] == entity_name:
            return index, entity["sent_id"]
    return None, None


# Process relations function
def process_relations(df):
    df["relations"] = None
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing rows for relations"
    ):
        relations = []
        relation_mentions = row["relationMentions"]
        vertex_set = row["vertexSet"]
        for relation in relation_mentions:
            h_name = relation["em1Text"]
            t_name = relation["em2Text"]
            r = relation["label"]
            h_idx, s_id = find_entity_index(vertex_set, h_name)
            t_idx, s_id_ = find_entity_index(vertex_set, t_name)
            if h_idx is not None and t_idx is not None:
                relations.append(
                    {
                        "h": h_idx,
                        "t": t_idx,
                        "r": r,
                        "evidence": list(set([s_id, s_id_])),
                    }
                )
        df.at[i, "relations"] = relations
    return df


# Process NERs function
def process_ners(df):
    def clean_word(word):
        return re.sub(r"[^\w\s]", "", word)

    labels = []
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing entities (vertexSet)"
    ):
        row_labels = []
        entity_mentions = row["entityMentions"]
        for entity_mention in entity_mentions:
            text = entity_mention["text"]
            words = text.split()
            for idx, sentence in enumerate(row["sents"]):
                pos = []
                for word in words:
                    clean_entity_word = clean_word(word)
                    found = False
                    for j, sent_word in enumerate(sentence):
                        if clean_word(sent_word) == clean_entity_word:
                            pos.append(j)
                            found = True
                    if not found:
                        print(f"Word '{word}' not found in sentence")
                        break
                if pos:
                    pos = [min(pos), max(pos) + 1]
                    row_labels.append(
                        {
                            "name": text,
                            "pos": pos,
                            "type": entity_mention["label"],
                            "sent_id": idx,
                        }
                    )
        labels.append(row_labels)
    df["vertexSet"] = labels
    return df


# Remove duplicates from dict function
def remove_duplicates_from_dict(data_dict):
    for key in data_dict:
        seen = set()
        unique_list = []
        for item in data_dict[key]:
            entity_tuple = (
                tuple(item["pos"]),
                item["type"],
                item["sent_id"],
                item["name"],
            )
            if entity_tuple not in seen:
                seen.add(entity_tuple)
                unique_list.append(item)
        data_dict[key] = unique_list
    return data_dict


# Fix grouped entities function
def fix_grouped_entities(combined_df):
    new_nyNER = np.empty(len(combined_df), dtype=object).tolist()
    new_relations = np.empty(len(combined_df), dtype=object).tolist()
    for i in tqdm(
        range(len(combined_df)),
        total=len(combined_df),
        desc="Processing rows - Fixing grouped entities",
    ):
        if combined_df.iloc[i]["org_dataset"] == "DocRED":
            row = combined_df.iloc[i]
            NER_DICT = {}
            for ner in row["NER"]:
                key = (ner["name"], ner["type"])
                if key not in NER_DICT:
                    NER_DICT[key] = [ner]
                else:
                    NER_DICT[key].append(ner)
            new_nyNER[i] = list(NER_DICT.values())
            new_relations[i] = combined_df.iloc[i]["labels"]
        if combined_df.iloc[i]["org_dataset"] != "DocRED":
            relations = combined_df.iloc[i]["labels"]
            NER = combined_df.iloc[i]["vertexSet"]
            nyNER = {}
            counter = 0
            tmp_dickies = {}
            for jdx, rel in enumerate(relations):
                e1 = NER[rel["h"]]
                e2 = NER[rel["t"]]
                if (e1["name"], e1["type"]) not in nyNER:
                    nyNER[(e1["name"], e1["type"])] = []
                    nyNER[(e1["name"], e1["type"])].append(e1)
                    tmp_dickies[(e1["name"], e1["type"])] = counter
                    counter += 1
                if (e2["name"], e2["type"]) not in nyNER:
                    nyNER[(e2["name"], e2["type"])] = []
                    nyNER[(e2["name"], e2["type"])].append(e2)
                    tmp_dickies[(e2["name"], e2["type"])] = counter
                    counter += 1
                relations[jdx]["h"] = tmp_dickies[(e1["name"], e1["type"])]
                relations[jdx]["t"] = tmp_dickies[(e2["name"], e2["type"])]
            for ner in NER:
                if (ner["name"], ner["type"]) not in nyNER.keys():
                    nyNER[(ner["name"], ner["type"])] = []
                    nyNER[(ner["name"], ner["type"])].append(ner)
            nyNER = remove_duplicates_from_dict(nyNER)
            new_nyNER[i] = list(nyNER.values())
            new_relations[i] = relations
    combined_df["nyVertexSet"] = new_nyNER
    combined_df["labels"] = new_relations
    combined_df["vertexSet"] = combined_df["nyVertexSet"]
    combined_df.drop(columns=["nyVertexSet"], inplace=True)
    print("Successfully fixed grouped entities")
    print("Columns in the dataframe: ", combined_df.columns)
    return combined_df


# Main function


def main(data_dir=None, file_no=-1, output_dir=None, download=True, verbose=True):

    if data_dir is None:
        data_dir = Path(root_folder, "data", "raw_data")
    nyt_data_dir = Path(data_dir, "NYT")

    if output_dir is None:
        output_dir = Path(data_dir, "NYT_Modified")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if download:
        download_and_manage_files(folder_id, nyt_data_dir)
    data_dir = str(nyt_data_dir / "*.json")
    json_files = glob.glob(data_dir)
    # if file_no == -1:
    dfs = []
    output_dir = Path(output_dir)
    print(Fore.RED + "Processing all files" + Style.RESET_ALL)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(json_files)):
        print(
            Fore.GREEN + f"Processing file {i + 1}/{len(json_files)}" + Style.RESET_ALL
        )
        df = pd.read_json(json_files[i], lines=True, orient="records")
        df["file_path"] = f"{json_files[i]}"
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    # else:
    #     df = pd.read_json(json_files[file_no], lines=True, orient="records")
    #     df["file_path"] = f"{json_files[file_no]}"
    #     output_dir.mkdir(parents=True, exist_ok=True)
    df = fix_sentences(df)
    df = process_ners(df)
    df = process_relations(df)
    df.drop(
        columns=["entityMentions", "sentText", "relationMentions", "sentId"],
        inplace=True,
    )
    df["org_dataset"] = "NYT"
    df["domains"] = "General"
    df = df.rename(columns={"articleId": "title", "relations": "labels"})
    df = fix_grouped_entities(df)

    df = save_json_with_progress(df, Path(output_dir, "NYT_modified.json"))
    print(
        f"{Fore.GREEN}Dataframe saved to {Path(output_dir, 'NYT_modified.json')}{Style.RESET_ALL}"
    )
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse NYT dataset.")
    parser.add_argument(
        "--data_dir", type=str, required=False, help="Location to match JSON files."
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
