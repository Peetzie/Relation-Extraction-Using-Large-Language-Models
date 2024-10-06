# REBEL_prepare.py
# Created by: Frederik Peetz-Schou Larsen
# Description: Script to download and process the CoNLL04 dataset.
# This script loads JSON files, processes named entity recognition (NER) and relation extraction data,
# and outputs the processed data to JSON files.
from multiprocessing import Pool
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import argparse
import re
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import json
import pandas as pd
import os
import os
import spacy
import string
from spacy.cli import download
from multiprocessing import Pool
from os import cpu_count
import pickle

sys.path.append(str(Path(__file__).resolve().parent))
from download_utils import *
import spacy


# %%
def load_NLP_model(root_dir):
    # Set up the cache directory from an environment variable or use a default
    cache_dir = f"{root_folder}/tmp"
    model_path = "en_core_web_sm"
    full_model_path = os.path.join(cache_dir, model_path)

    # Try to load the SpaCy model from the specified directory
    try:
        global nlp
        nlp = spacy.load(full_model_path)

    except OSError:
        # If the model is not found, download it and save to the specified directory
        print("Model not found. Downloading now...")
        download(model_path)  # This downloads the model to the default SpaCy path
        nlp = spacy.load(model_path)  # Load model from the default location
        nlp.to_disk(full_model_path)  # Save the model to your custom directory
        print(f"Model downloaded and saved to {full_model_path}")

    # Example usage of the loaded model
    doc = nlp("This is an example sentence. This is another one.")
    sentences = [[sent.text] for sent in doc.sents]
    print(sentences)


def split_text_into_sentences(text):
    doc = nlp(text)
    return [[sent.text] for sent in doc.sents]


def process_row(row):
    text = row["text"]
    sents = split_text_into_sentences(text)
    return sents


def apply_multiprocessing(data_frame):
    # Create a pool of processes. You can specify the number of processes or use the number of CPUs
    with Pool(processes=cpu_count()) as pool:
        # Convert DataFrame rows to a list of tuples (index, row) for processing
        row_list = [row for index, row in data_frame.iterrows()]

        # Use imap for lazy iteration and wrap it with tqdm for the progress bar
        results = list(
            tqdm(
                pool.imap(process_row, row_list),
                total=len(row_list),
                desc="Processing Rows",
            )
        )

    # Assign results to the 'sents' column
    data_frame["sents"] = results
    return data_frame


def load_data(base_directory):
    # Define the path to the directory containing the .jsonl files
    directory_path = Path(base_directory) / "data" / "rebel_dataset"
    target_directory = Path(base_directory)

    # Move all .jsonl fils
    files_to_move = directory_path.glob("*jsonl")
    for file_path in files_to_move:
        shutil.move(str(file_path), target_directory / file_path.name)

    # Check if the source_directory is empty and remove it if so
    if not list(directory_path.iterdir()):  # Check if directory is empty
        directory_path.rmdir()  # Remove the source directory
    else:
        print("Rebel dataset directory not empty, not removed.")

    # Check if the parent directory of source_directory (data directory) is empty and remove it if so
    parent_directory = directory_path.parent
    if not list(parent_directory.iterdir()):  # Check if directory is empty
        parent_directory.rmdir()  # Remove the data directory
    else:
        print("Data directory not empty, not removed.")

    # Collect all .jsonl files in the directory
    files = [f for f in target_directory.glob("*.jsonl")]
    # files = [files[0]]
    data_frames = []

    # Process each .jsonl file
    for i, file_path in enumerate(files):
        # First, determine the number of lines to set tqdm total
        total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

        json_objects = []
        with open(file_path, "r", encoding="utf-8") as file:
            # Use total_lines to correctly display progress
            progress_bar = tqdm(
                file, total=total_lines, desc=f"Processing {file_path.name}"
            )
            for line in progress_bar:
                json_objects.append(json.loads(line.strip()))

        # Convert list of dictionaries to DataFrame
        df = pd.json_normalize(json_objects)

        # Add the additional columns
        df["org_dataset"] = "REBEL"
        df["file_path"] = str(file_path)
        df["domains"] = "General"

        data_frames.append(df)

        # Clear memory
        del json_objects, df

    # Concatenate all DataFrames into one
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


def process_data_from_df(df):
    unique_q_uris = set()
    predicate_properties = set()

    # Iterate over each row in the DataFrame
    for index, row in tqdm(
        df.iterrows(),
        desc="Checking types of entities and relations",
        total=df.shape[0],
    ):
        # Access the entities and triples directly from the DataFrame columns
        entities = row["entities"]
        triples = row["triples"]

        # Process entities to capture Q URIs
        for entity in entities:
            uri = entity.get("uri", "")
            if uri.startswith("Q"):
                unique_q_uris.add(uri)

        # Process triples to capture P URIs
        for triple in triples:
            uri = triple.get("predicate", {}).get("uri", "")
            if uri.startswith("P"):
                predicate_properties.add(uri)

    return list(unique_q_uris), list(predicate_properties)


def fetch_relations_types(property_ids):
    """
    Fetch the titles of multiple Wikidata properties given their property IDs using SPARQL.

    Args:
    property_ids (list): List of property IDs (e.g., ['P31', 'P2046']).

    Returns:
    dict: A dictionary mapping property IDs to their titles.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    results_dict = {}

    # Split the IDs into batches to manage request size
    batch_size = 50
    for i in tqdm(
        range(0, len(property_ids), batch_size), desc="Fetching relations types"
    ):
        batch = property_ids[i : i + batch_size]
        ids_str = " ".join(f"wd:{pid}" for pid in batch)
        query = f"""
        SELECT ?property ?propertyLabel WHERE {{
          VALUES ?property {{ {ids_str} }}
          ?property rdfs:label ?propertyLabel.
          FILTER (LANG(?propertyLabel) = "en")
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                pid = result["property"]["value"].split("/")[-1]
                results_dict[pid] = result["propertyLabel"]["value"]
        except Exception as e:
            print(f"An error occurred: {e}")

    return results_dict


def check_and_remove_qids(df):
    """
    Check for QXXXXX values in the dataset and remove rows containing them.

    Args:
    df (pandas.DataFrame): The DataFrame to check and clean.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with no QXXXXX values.
    list: List of QXXXXX values found and removed.
    """
    qid_pattern = re.compile(r"^Q\d+$")
    qids_found = []

    # Check vertexSet for QXXXXX values
    for idx, vertex_list in df["vertexSet"].items():
        for entity_list in vertex_list:
            for entity in entity_list:
                if qid_pattern.match(entity["type"]):
                    qids_found.append(entity["type"])

    # Check labels for QXXXXX values
    for idx, labels in df["labels"].items():
        for relation in labels:
            if qid_pattern.match(relation["r"]):
                qids_found.append(relation["r"])

    # Remove rows containing QXXXXX values
    qids_set = set(qids_found)
    df_cleaned = df.copy()

    def remove_qids_from_vertex_set(vertex_set):
        return [
            [entity for entity in entity_list if entity["type"] not in qids_set]
            for entity_list in vertex_set
        ]

    df_cleaned["vertexSet"] = df_cleaned["vertexSet"].apply(remove_qids_from_vertex_set)
    df_cleaned = df_cleaned[df_cleaned["vertexSet"].map(len) != 0]

    def remove_qids_from_labels(labels):
        return [relation for relation in labels if relation["r"] not in qids_set]

    df_cleaned["labels"] = df_cleaned["labels"].apply(remove_qids_from_labels)
    df_cleaned = df_cleaned[df_cleaned["labels"].map(len) != 0]

    return df_cleaned, list(set(qids_found))


# def split_text_into_sentences(text):
#     """
#     Splits the given text into a list of sentences.

#     Args:
#         text (str): The input text to be split into sentences.

#     Returns:
#         list: A list of sentences.
#     """
#     # Split text into sentences
#     sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

#     # Remove any empty strings resulting from the split
#     sentences = [[sentence] for sentence in sentences if sentence]

#     return sentences


import re


def process_vertex_set(df):
    df["vertexSet"] = None

    def preprocess_text(text):
        # Normalize spaces and lowercase the text for processing
        return " ".join(text.split()).lower()

    def find_phrase_in_sentences(sentences, phrase):
        phrase_words = preprocess_text(phrase).split()
        phrase_length = len(phrase_words)

        for sentence_idx, sentence in enumerate(sentences):
            processed_sentence = [preprocess_text(word) for word in sentence]
            for start_idx in range(len(processed_sentence) - phrase_length + 1):
                current_slice = processed_sentence[
                    start_idx : start_idx + phrase_length
                ]
                if current_slice == phrase_words:
                    return sentence_idx, start_idx, start_idx + phrase_length
        return -1, -1, -1

    def secondary_pass(sentences, entity):
        # Simplified check to see if entity is a substring of any word in the sentence
        entity_lower = preprocess_text(entity)
        for sentence_idx, sentence in enumerate(sentences):
            for word_idx, word in enumerate(sentence):
                if entity_lower in preprocess_text(word):
                    return sentence_idx, word_idx, word_idx + 1
        return -1, -1, -1

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing vertexSet"):
        vertexSet = []
        entities = row["entities"]
        sents = row["sents_split"]
        missed_entities = []

        for entity in entities:
            sent_id, idx_start, idx_end = find_phrase_in_sentences(
                sents, entity["surfaceform"]
            )
            if sent_id == -1:
                missed_entities.append(entity)
            else:
                vertexSet.append(
                    {
                        "name": entity["surfaceform"],
                        "pos": [idx_start, idx_end],  # Adjust end index to be inclusive
                        "type": entity["uri"],
                        "sent_id": sent_id,
                    }
                )

        # Secondary pass for missed entities
        for entity in missed_entities:
            sent_id, idx_start, idx_end = secondary_pass(sents, entity["surfaceform"])
            if sent_id != -1:
                vertexSet.append(
                    {
                        "name": entity["surfaceform"],
                        "pos": [idx_start, idx_end],  # Adjust to be inclusive
                        "type": entity["uri"],
                        "sent_id": sent_id,
                    }
                )

        df.at[idx, "vertexSet"] = vertexSet

    return df


def save_to_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


# Function to update entities
def update_entities(entities, QID_Mapping_dict):
    for entity in entities:
        if entity["uri"] in QID_Mapping_dict:
            entity["uri"] = QID_Mapping_dict[entity["uri"]]
    return entities


# Function to update triples
def update_triples(triples, property_titles):
    for triple in triples:
        if triple["predicate"]["uri"] in property_titles:
            triple["predicate"]["uri"] = property_titles[triple["predicate"]["uri"]]
    return triples


def find_entity_index_and_sent_id(vertex_set, entity):
    for i, sublist in enumerate(vertex_set):
        for item in sublist:
            if item["name"] == entity["surfaceform"] and item["type"] == entity["uri"]:
                return i, item["sent_id"]
    return -1, -1  # If entity is not found


def build_relations(df):
    df["labels"] = None
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building relations"):
        triples = row["triples"]
        vertex_set = row["vertexSet"]
        relations = []

        for triple in triples:
            relation = triple["predicate"]
            r = relation["uri"]
            subject = triple["subject"]
            obj = triple["object"]

            subject_index, subject_sent_id = find_entity_index_and_sent_id(
                vertex_set, subject
            )
            object_index, object_sent_id = find_entity_index_and_sent_id(
                vertex_set, obj
            )

            if subject_index != -1 and object_index != -1:
                relations.append(
                    {
                        "h": subject_index,
                        "t": object_index,
                        "r": r,
                        "evidence": list(set([subject_sent_id, object_sent_id])),
                    }
                )

        df.at[idx, "labels"] = relations
    return df


import numpy as np


def remove_duplicates_from_dict(data_dict):
    """
    TODO Duplication code.. Fix it later
    """
    for key in data_dict:
        seen = set()
        unique_list = []
        for item in data_dict[key]:
            # Create a tuple of the relevant parameters
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


def fix_grouped_entities(combined_df):
    """
    TODO
    Fix grouped entities in the combined DataFrame.
    This code is gargbae..It is copy from the combiner.. However it works. Deal with it later..
    """
    # Assuming combined_df is already defined and loaded with data
    new_nyNER = np.empty(len(combined_df), dtype=object).tolist()
    new_relations = np.empty(len(combined_df), dtype=object).tolist()
    # Process each row
    for i in tqdm(
        range(len(combined_df)),
        total=len(combined_df),
        desc="Processing rows - Fixing grouped entities",
    ):
        if (
            combined_df.iloc[i]["org_dataset"] == "DocRED"
            or combined_df.iloc[i]["org_dataset"] == "REBEL"
        ):
            row = combined_df.iloc[i]
            # Initialize a dictionary to store the NER data for the current row
            NER_DICT = {}
            # Iterate through each NER entry in the row's 'NER' column
            for ner in row["vertexSet"]:
                # Create a unique key for each name, type combination

                key = (ner["name"], ner["type"])

                # Append the entity to the list for this key in NER_DICT
                if key not in NER_DICT:
                    NER_DICT[key] = [ner]
                else:
                    NER_DICT[key].append(ner)

            # Update the 'nyNER' column in the DataFrame with the populated dictionary values
            new_nyNER[i] = list(NER_DICT.values())
            if combined_df.iloc[i]["org_dataset"] == "DocRED":
                new_relations[i] = combined_df.iloc[i][
                    "labels"
                ]  # Append original relations for 'DocRED'
            else:
                new_relations[i] = None  ## Will be fixed later
        else:
            relations = combined_df.iloc[i]["labels"]
            NER = combined_df.iloc[i]["vertexSet"]
            nyNER = {}  # Initialize the dictionary for this row
            counter = 0
            tmp_dickies = {}

            for jdx, rel in enumerate(relations):
                e1 = NER[rel["h"]]
                e2 = NER[rel["t"]]

                # Check if the tuple (e1 name, e1 type) exists as a key, if not, initialize as empty list

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

                # Update indices for relations
                relations[jdx]["h"] = tmp_dickies[(e1["name"], e1["type"])]
                relations[jdx]["t"] = tmp_dickies[(e2["name"], e2["type"])]

            for ner in NER:
                if (ner["name"], ner["type"]) not in nyNER.keys():
                    nyNER[(ner["name"], ner["type"])] = []
                    nyNER[(ner["name"], ner["type"])].append(ner)

            # Remove duplicates from the dictionary
            nyNER = remove_duplicates_from_dict(nyNER)

            new_nyNER[i] = list(nyNER.values())
            new_relations[i] = relations

    # Update the original DataFrame
    combined_df["nyVertexSet"] = new_nyNER
    combined_df["labels"] = new_relations
    combined_df["vertexSet"] = combined_df["nyVertexSet"]

    # Drop the column "nyNER"
    combined_df.drop(columns=["nyVertexSet"], inplace=True)
    print("Successfully fixed grouped entities")
    print("Columns in the dataframe: ", combined_df.columns)
    return combined_df


def tokenize_sentences(df):
    df["sents_split"] = None
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc="Splitting sentences into words"
    ):
        sents = row["sents"]
        processed = [sentence[0].split(" ") for sentence in sents]
        df.at[i, "sents_split"] = processed
    return df


def clean_invalid_entities(df):
    total_entities_removed = 0
    total_relations_removed = 0
    total_evidence_removed = 0

    for index, row in df.iterrows():
        entities_to_remove = set()

        # Identify invalid entities
        for entity_list in row["vertexSet"]:
            for item in entity_list:
                if (
                    item["pos"][0] == -1
                    or item["pos"][1] == -1
                    or item["sent_id"] == -1
                ):
                    entities_to_remove.add(item["name"])

        # Remove invalid entities
        initial_entity_count = sum(len(entity_list) for entity_list in row["vertexSet"])
        row["vertexSet"] = [
            entity_list
            for entity_list in row["vertexSet"]
            if entity_list[0]["name"] not in entities_to_remove
        ]
        final_entity_count = sum(len(entity_list) for entity_list in row["vertexSet"])
        total_entities_removed += initial_entity_count - final_entity_count

        # Remove relations involving invalid entities or invalid evidence
        if "labels" in row:
            initial_relation_count = len(row["labels"])
            valid_relations = []
            for relation in row["labels"]:
                head_idx = relation["h"]
                tail_idx = relation["t"]
                if head_idx < len(row["vertexSet"]) and tail_idx < len(
                    row["vertexSet"]
                ):
                    head_entity = row["vertexSet"][head_idx][0]
                    tail_entity = row["vertexSet"][tail_idx][0]
                    if (
                        head_entity["name"] not in entities_to_remove
                        and tail_entity["name"] not in entities_to_remove
                    ):
                        # Remove invalid evidence
                        initial_evidence_count = len(relation["evidence"])
                        relation["evidence"] = [
                            e for e in relation["evidence"] if e != -1
                        ]
                        final_evidence_count = len(relation["evidence"])
                        total_evidence_removed += (
                            initial_evidence_count - final_evidence_count
                        )
                        if relation["evidence"]:
                            valid_relations.append(relation)
            row["labels"] = valid_relations
            final_relation_count = len(row["labels"])
            total_relations_removed += initial_relation_count - final_relation_count

    print(f"Total entities removed: {total_entities_removed}")
    print(f"Total relations removed: {total_relations_removed}")
    print(f"Total evidence removed: {total_evidence_removed}")

    return df


def update_vertexSet(row, entity_mappings):
    vertexSet = row["vertexSet"]
    for entity_list in vertexSet:
        for entity in entity_list:
            entity_type = entity["type"]
            if entity_type.startswith("Q"):
                entity["type"] = entity_mappings.get(entity_type, entity_type)
            elif "http://www.w3.org" in entity_type:
                # Extract the last part of the string after the last '#'
                new_type = entity_type.split("#")[-1]
                entity["type"] = new_type.capitalize()
    return vertexSet


def remove_rows_with_qids(df, qids):
    """
    Remove rows from a DataFrame where the entity has a Q-value defined in qids.

    Args:
    df (pandas.DataFrame): The DataFrame to remove rows from.
    qids (list): List of Q-values to check against.

    Returns:
    pandas.DataFrame: The modified DataFrame with rows removed.
    """
    qid_set = set(qids)  # Convert list to set for faster lookups
    tqdm.pandas(desc="Removing rows")

    def check_qids(entities):
        for entity in entities:
            if entity.get("uri", "") in qid_set:
                return False
        return True

    filtered_df = df[df["entities"].progress_apply(check_qids)]
    return filtered_df


def cap(df):
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Adjusting capitalizations"):
        relations = row["labels"]
        for rel in relations:
            rel["r"] = rel["r"].capitalize()
        for vertex in row["vertexSet"]:
            for entity in vertex:
                entity["type"] = entity["type"].capitalize()
    return df


def verify_entities(df):
    empty_entities = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying entities"):
        for entity_list in row["vertexSet"]:
            for entity in entity_list:
                if len(entity) == 0:  # constraint from verification assert
                    empty_entities.append(idx)
    return empty_entities


def main(
    data_dir=None,
    file_no=-1,
    output_dir=None,
    download=True,
    verbose=True,
    parse_entities=False,
):

    if data_dir is None:
        data_dir = Path(root_folder, "data", "raw_data")
    REBEL_data_dir = Path(data_dir, "REBEL")
    if not os.path.exists(REBEL_data_dir):
        os.makedirs(REBEL_data_dir)

    if output_dir is None:
        output_dir = Path(data_dir, "REBEL_Modified")
    extract_dir = Path(REBEL_data_dir, "extracted_files")
    if download:
        print("Downloading data")
        download_url = (
            "https://osf.io/download/96fpt/?view_only=87e7af84c0564bd1b3eadff23e4b7e54"
        )
        downloader_dir = Path(REBEL_data_dir, "downloaded.zip")
        download_zip(download_url, downloader_dir)
        extract_zip(downloader_dir, extract_dir)
        # Set up the cache directory from an environment variable or use a default
    load_NLP_model(root_folder)
    df = load_data(extract_dir)
    # Check if the file exists

    if parse_entities:
        print("Using pre-defined entities ")
        entities_mapping_file = "/work3/s174159/LLM_Thesis/databuilding/types/REBEL/Entities/Entities_mapping.pkl"
        missing_qids = "/work3/s174159/LLM_Thesis/databuilding/types/REBEL/Entities/missing_qids.pkl"

        file_list = [entities_mapping_file, missing_qids]
        if all([os.path.isfile(f) for f in file_list]):
            with open(entities_mapping_file, "rb") as f:
                entities_mapping = pickle.load(f)
            print("Loaded entities")

            with open(missing_qids, "rb") as f:
                filtered_q_uris = pickle.load(f)
            print("Loaded filtered")
            print("All relation and entity mapping files loaded correctly")

    unique_entity_types, unique_relations = process_data_from_df(df)
    relation_mappings = fetch_relations_types(unique_relations)

    ## Applying relations
    tqdm.pandas(desc="Updating triples with properties mapping")
    df["triples"] = df["triples"].progress_apply(
        update_triples, property_titles=relation_mappings
    )
    print("relation mapping", relation_mappings)

    # Checking parsing
    unique_entity_types, unique_relations = process_data_from_df(df)
    unique_relations = [
        relation_type
        for relation_type in unique_relations
        if re.match(r"^Q\d+$", relation_type)
    ]
    assert (
        len(unique_relations) == 0
    ), f"Relation types not parsed. Exiting. {unique_relations}"

    # Apply multiprocessing to the DataFrame
    df = apply_multiprocessing(df)

    df = tokenize_sentences(df)

    df = process_vertex_set(df)
    df["org_dataset"] = "REBEL"
    df = fix_grouped_entities(df)
    df = build_relations(df)

    if parse_entities:
        print("Using pre-defined entities ")
        entities_mapping_file = (
            f"{root_folder}/databuilding/types/REBEL/Entities/Entities_mapping.pkl"
        )
        missing_qids = (
            f"{root_folder}/databuilding/types/REBEL/Entities/missing_qids.pkl"
        )

        file_list = [entities_mapping_file, missing_qids]
        if all([os.path.isfile(f) for f in file_list]):
            with open(entities_mapping_file, "rb") as f:
                entities_mapping = pickle.load(f)
            print("Loaded entities")

            with open(missing_qids, "rb") as f:
                filtered_q_uris = pickle.load(f)
            print("Loaded filtered")
            print("All relation and entity mapping files loaded correctly")
        # Apply the function to the DataFrame with a total progress bar for the number of rows
        for i in tqdm(range(len(df)), desc="Processing rows"):
            df.at[i, "vertexSet"] = update_vertexSet(df.iloc[i], entities_mapping)
    df = df.drop(columns=["sents"])
    df = df.rename(columns={"sents_split": "sents"})
    df = df.drop(columns=["triples", "entities", "text", "docid"])
    df = cap(df)
    ## Check empty vertexSet column
    # Count rows where 'VertexSet' is an empty list
    empty_count = (df["vertexSet"].map(len) == 0).sum()
    print(f"Number of rows with 'VertexSet' as empty list: {empty_count}")

    # Drop rows where 'VertexSet' is an empty list
    df = df[df["vertexSet"].map(len) != 0]
    df = clean_invalid_entities(df)
    df = df[
        ["org_dataset", "title", "sents", "vertexSet", "labels", "file_path", "domains"]
    ]
    if parse_entities:
        # Check for any remaining QXXXXX values, remove them, and output the list
        df, qids_removed = check_and_remove_qids(df)
        if qids_removed:
            print(f"QIDs removed: {qids_removed}")
            print(f"Number of QIDs removed: {len(qids_removed)}")

        empty_entities = verify_entities(df)
        if len(empty_entities) > 0:
            print(f"# of empty entities to be removed: {len(empty_entities)}")
            print(f"Found empty entities in rows: {empty_entities}")
            for index, _ in empty_entities:
                df.drop(index, inplace=True)
    # Save dataframe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "REBEL_processed.json")
    save_json_with_progress(df, output_file)
    print(f"Data has been processed and saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sciERC dataset.")

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
    parser.add_argument(
        "--parse_entities",
        action="store_true",
        help="Remove not found entities -- Only possible with the pickle files.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.file_no,
        args.output_dir,
        args.download,
        args.verbose,
        args.parse_entities,
    )
