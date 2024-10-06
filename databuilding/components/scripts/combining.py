import argparse
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from colorama import Fore, Style, init
import glob
import json

sys.path.append(str(Path(__file__).resolve().parent))
from download_utils import *


def combining_df(folders, verbose=True, chunk_size=1000):
    dataframes = []
    for folder in folders:
        json_files = glob.glob(folder + "/*.json")
        json_files = [x for x in json_files if "rel_info" not in x]

        for file in tqdm(
            json_files, desc=f"Processing files in {folder}", disable=not verbose
        ):
            # Open the file to count lines
            with open(file, "r") as f:
                total_size = sum(1 for line in f)

            # Set up progress bar based on the file's total size
            with tqdm(
                total=total_size,
                desc=f"Loading {file}",
                leave=False,
                disable=not verbose,
            ) as pbar:
                # Read JSON in chunks
                json_reader = pd.read_json(
                    file, orient="records", lines=True, chunksize=chunk_size
                )
                for chunk in json_reader:
                    chunk["original_file_path"] = file  # Rename directly here
                    dataframes.append(chunk)
                    pbar.update(len(chunk))  # Update with actual chunk size

    combined_df = pd.concat(dataframes)
    combined_df.rename(
        columns={"file_path": "original_file_path", "filename": "original_file_path"},
        inplace=True,
    )
    return combined_df


def remove_empty_entities(df, verbose=True):
    invalid_indices = []
    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking for empty entity lists",
        disable=not verbose,
    ):
        entities = row["vertexSet"]
        for e in entities:
            if len(e) == 0:
                invalid_indices.append(i)
                break  # No need to check further for this row

    num_removed = len(invalid_indices)
    if num_removed > 0:
        df = df.drop(invalid_indices).reset_index(drop=True)

    print(f"Number of rows removed due to empty entity lists: {num_removed}")
    return df


def get_ent_rel_lists(combined_df, output_dir, dataset_name=None):
    entities_lst = []
    relations_lst = []
    for i, row in tqdm(
        combined_df.iterrows(),
        total=len(combined_df),
        desc="Getting unique entities and relations",
    ):
        # print(row)
        entities = row["vertexSet"]
        relations = row["labels"]
        for ent_lst in entities:
            for ent in ent_lst:
                entities_lst.append(ent["type"])
        for rel in relations:
            relations_lst.append(rel["r"])
    ent_set = set(entities_lst)
    rel_set = set(relations_lst)

    print(f"Unique entities: {len(ent_set)}, Unique relations: {len(rel_set)}")

    if len(ent_set) > 200:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        entity_dict = {}
        relation_dict = {}
        for rel in rel_set:
            relation_dict[rel] = rel
        for ent in ent_set:
            entity_dict[ent] = ent

        with open(Path(output_path, "entities.json"), "w") as outfile:
            json.dump(entity_dict, outfile)
        with open(Path(output_path, "relations.json"), "w") as outfile:
            json.dump(relation_dict, outfile)

        print(
            "Successfully saved unique entities and relations to JSON files -- Remember to check for duplicates"
        )
    # Ensure output directory exists
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use ExcelWriter to write to multiple sheets
        with pd.ExcelWriter(output_path / "types.xlsx") as writer:
            pd.DataFrame(list(ent_set), columns=["Entities"]).to_excel(
                writer, sheet_name="Entities", index=False
            )
            pd.DataFrame(list(rel_set), columns=["Relations"]).to_excel(
                writer, sheet_name="Relations", index=False
            )
        print(
            "Successfully saved unique entities and relations to Excel file -- Remember to check for duplicates"
        )


def remove_duplicates_from_dict(data_dict):
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


# def fix_grouped_entities(combined_df, verbose):
#     # Assuming combined_df is already defined and loaded with data
#     new_nyNER = np.empty(len(combined_df), dtype=object).tolist()
#     new_relations = np.empty(len(combined_df), dtype=object).tolist()
#     # Process each row
#     for i in tqdm(
#         range(len(combined_df)),
#         total=len(combined_df),
#         desc="Processing rows - Fixing grouped entities",
#     ):
#         if (
#             combined_df.iloc[i]["org_dataset"] == "DocRED_Distant"
#             or combined_df.iloc[i]["org_dataset"] == "DocRED_Joint"
#             or combined_df.iloc[i]["org_dataset"] == "REBEL"
#         ):
#             print("REBEL")
#             row = combined_df.iloc[i]
#             # Initialize a dictionary to store the NER data for the current row
#             NER_DICT = {}

#             # Iterate through each NER entry in the row's 'NER' column
#             for ner in row["vertexSet"]:
#                 # Create a unique key for each name, type combination
#                 key = (ner["name"], ner["type"])

#                 # Append the entity to the list for this key in NER_DICT
#                 if key not in NER_DICT:
#                     NER_DICT[key] = [ner]
#                 else:
#                     NER_DICT[key].append(ner)

#             # Update the 'nyNER' column in the DataFrame with the populated dictionary values
#             new_nyNER[i] = list(NER_DICT.values())
#             new_relations[i] = combined_df.iloc[i][
#                 "labels"
#             ]  # Append original relations for 'DocRED'
#         else:
#             print("Else")
#             relations = combined_df.iloc[i]["labels"]
#             NER = combined_df.iloc[i]["vertexSet"]
#             nyNER = {}  # Initialize the dictionary for this row
#             counter = 0
#             tmp_dickies = {}

#             for jdx, rel in enumerate(relations):
#                 e1 = NER[rel["h"]]
#                 e2 = NER[rel["t"]]

#                 # Check if the tuple (e1 name, e1 type) exists as a key, if not, initialize as empty list
#                 if (e1["name"], e1["type"]) not in nyNER:
#                     nyNER[(e1["name"], e1["type"])] = []
#                     nyNER[(e1["name"], e1["type"])].append(e1)
#                     tmp_dickies[(e1["name"], e1["type"])] = counter
#                     counter += 1

#                 if (e2["name"], e2["type"]) not in nyNER:
#                     nyNER[(e2["name"], e2["type"])] = []
#                     nyNER[(e2["name"], e2["type"])].append(e2)
#                     tmp_dickies[(e2["name"], e2["type"])] = counter
#                     counter += 1

#                 # Update indices for relations
#                 relations[jdx]["h"] = tmp_dickies[(e1["name"], e1["type"])]
#                 relations[jdx]["t"] = tmp_dickies[(e2["name"], e2["type"])]

#             for ner in NER:
#                 if (ner["name"], ner["type"]) not in nyNER.keys():
#                     nyNER[(ner["name"], ner["type"])] = []
#                     nyNER[(ner["name"], ner["type"])].append(ner)

#             # Remove duplicates from the dictionary
#             nyNER = remove_duplicates_from_dict(nyNER)

#             new_nyNER[i] = list(nyNER.values())
#             new_relations[i] = relations

#     # Update the original DataFrame
#     combined_df["nyVertexSet"] = new_nyNER
#     combined_df["labels"] = new_relations
#     combined_df["vertexSet"] = combined_df["nyVertexSet"]

#     # Drop the column "nyNER"
#     combined_df.drop(columns=["nyVertexSet"], inplace=True)
#     print("Successfully fixed grouped entities")
#     print("Columns in the dataframe: ", combined_df.columns)
#     return combined_df


def save_df_to_formats(df, output_dir):
    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()]
    print("Duplicate columns:", duplicate_columns)
    # If there are duplicates, remove them
    df = df.loc[:, ~df.columns.duplicated()]
    save_json_with_progress(df, Path(output_dir, "combined_dataset.json"))
    print("Successfully saved all datasets")


def split_distant(df):
    # Splitting into train_distant dataset
    train_distant = df[df["original_file_path"].str.contains("distant")]
    # Removing rows from combined_df
    df = df[~df["original_file_path"].str.contains("distant")]
    return train_distant, df


def main(folders, output_dir=None, verbose=True):
    root_dir = Path(root_folder)
    if output_dir is None:
        output_dir = Path(root_dir, "pre-processed data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = combining_df(folders, verbose)
    get_ent_rel_lists(df, output_dir)

    df = remove_empty_entities(df, verbose)
    save_df_to_formats(df, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and pre-process the dataset")
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="List of folders containing datasets to combine",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The directory to save the processed data.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity."
    )
    args = parser.parse_args()
    main(
        args.folders,
        args.output_dir,
        args.verbose,
    )
