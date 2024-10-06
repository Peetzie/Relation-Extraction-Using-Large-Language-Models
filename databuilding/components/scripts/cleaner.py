import sys
import os
from pathlib import Path
from colorama import Style
from tqdm import tqdm
import pandas as pd
import json
import argparse

sys.path.append(str(Path(__file__).resolve().parent))
from download_utils import *


def original_statistics(df, output_dir=None):
    total_no_of_sentences = 0
    total_no_relations = 0
    total_no_entities = 0

    types_entities = []
    types_relations = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Calculating statistics"):
        for sentence in row["sents"]:
            total_no_of_sentences += 1

        for relation in row["labels"]:
            total_no_relations += 1
            types_relations.append(relation["r"])
        for entity in row["vertexSet"]:
            for ent in entity:
                total_no_entities += 1
                try:
                    types_entities.append(ent["type"])
                except:
                    print("error appending entity type")

    print("Total number of sentences: ", total_no_of_sentences)
    print("Total number of relations: ", total_no_relations)
    print("Total number of entities: ", total_no_entities)
    print("There are ", len(set(types_entities)), " unique entity types")
    print("There are ", len(set(types_relations)), " unique relation types")

    if output_dir:
        # Create a DataFrame with the statistics
        statistics_df = pd.DataFrame(
            {
                "Total number of sentences": [total_no_of_sentences],
                "Total number of relations": [total_no_relations],
                "Total number of entities": [total_no_entities],
                "Number of unique entity types": [len(set(types_entities))],
                "Number of unique relation types": [len(set(types_relations))],
            }
        )

        # Save the DataFrame as a CSV file
        statistics_df.to_csv(Path(output_dir, "statistics.csv"), index=False)
    return (
        total_no_of_sentences,
        total_no_relations,
        total_no_entities,
        len(set(types_entities)),
        len(set(types_relations)),
    )


def update_types(combined_df, processed_data_dir, input_dir=None):
    # Load relation types with fallback

    rel_path = Path(processed_data_dir, "types.xlsx")
    print("input_file", input_dir)
    print("Reading excel", rel_path)
    relation_types_df = pd.read_excel(rel_path, "Relations", engine="openpyxl")

    if "Cleaned Relations" in list(relation_types_df.columns):
        rel_types = relation_types_df.set_index("Relations").to_dict()[
            "Cleaned Relations"
        ]
    else:
        rel_types = relation_types_df.set_index(
            "Relations"
        ).to_dict()  # Fallback to original

    # Load entity types with fallback
    entity_types_df = pd.read_excel(rel_path, "Entities")
    if "Cleaned Entities" in list(entity_types_df.columns):
        ent_types = entity_types_df.set_index("Entities").to_dict()["Cleaned Entities"]
    else:
        ent_types = entity_types_df.set_index(
            "Entities"
        ).to_dict()  # Fallback to original

    # Update DataFrame with new types
    for i, row in tqdm(
        combined_df.iterrows(), total=len(combined_df), desc="Updating types"
    ):
        for rel in row["labels"]:
            rel["r"] = rel_types.get(
                rel["r"], rel["r"]
            )  # Fallback to original if not found in dict
        for entities in row["vertexSet"]:
            for ent in entities:
                ent["type"] = ent_types.get(
                    ent["type"], ent["type"]
                )  # Fallback to original if not found in dict

    return combined_df


def load_json_with_progress(file_path, chunk_size=1000):
    # Calculate the total number of lines to set the tqdm total
    total_lines = sum(1 for line in open(file_path, "r", encoding="utf-8"))

    # Initialize the progress bar
    pbar = tqdm(total=total_lines, desc=f"Loading {file_path}")

    # Read the JSON file in chunks
    json_reader = pd.read_json(
        file_path, orient="records", lines=True, chunksize=chunk_size
    )

    # Initialize an empty list to hold dataframes
    dataframes = []

    # Process each chunk
    for chunk in json_reader:
        dataframes.append(chunk)
        # Update the progress bar by the size of the chunk
        pbar.update(len(chunk))

    # Close the progress bar after completing the loop
    pbar.close()

    # Concatenate all chunks into a single dataframe
    combined_df = pd.concat(dataframes)
    return combined_df


def main(input_dir, output_dir):
    # Construct the path to the combined dataset JSON file within the input directory
    combined_json_path = Path(input_dir) / "combined_dataset.json"

    # Load combined dataset
    combined_df = load_json_with_progress(combined_json_path, chunk_size=1000)

    # Calculate original statistics
    original_statistics(combined_df, output_dir)

    # Update types ONLY IF NOT REBEL
    if "REBEL" not in output_dir:
        if input_dir is None:
            combined_df = update_types(combined_df, output_dir)
        else:
            combined_df = update_types(combined_df, output_dir, combined_json_path)
        # Calculate updated statistics
        original_statistics(combined_df, output_dir)
    else:
        print(
            f"{Fore.CYAN}CLEANING REBEL NOT REQUIRED - SKIPPED. FINALIZING SAVING IN CORRECT FORMAT.{Style.RESET_ALL}"
        )
    # Save updated dataset
    save_df_no_lines_json(combined_df, output_dir)
    # combined_df.to_json(
    #     Path(output_dir, "combined_dataset_corrected.json"),
    #     orient="records",
    #     lines=True,
    # )
    print(
        "Updated combined dataset saved to",
        Path(output_dir, "combined_dataset_corrected.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update types in combined dataset and calculate statistics"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory that stores all the dataset files",
    )
    args = parser.parse_args()

    main(args.output_dir)
