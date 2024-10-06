import argparse
import pandas as pd
from pathlib import Path
import json


def load_excel(excel_file):
    xl = pd.ExcelFile(excel_file)

    # Read sheets
    df_entities_sheet = xl.parse("Entities")
    df_relations_sheet = xl.parse("Relations")

    # Determine the column names to use for entities and relations
    entity_col = (
        "Cleaned Entities"
        if "Cleaned Entities" in df_entities_sheet.columns
        else "Entities"
    )
    relation_col = (
        "Cleaned Relations"
        if "Cleaned Relations" in df_relations_sheet.columns
        else "Relations"
    )

    # Get unique entities and relations
    unique_entities = df_entities_sheet[entity_col].dropna().unique()
    unique_relations = df_relations_sheet[relation_col].dropna().unique()

    return unique_entities, unique_relations


def load_json(json_file):
    with open(json_file, "r") as j_file:
        data = json.load(j_file)
    return data


def main(input_dir, output_dir):
    excel_entities = []
    excel_relations = []

    json_entities = []
    json_relations = []

    input_dir = Path(input_dir)

    excel_file_path = input_dir / "types.xlsx"
    entities_json_path = input_dir / "entities.json"
    relations_json_path = input_dir / "relations.json"

    if excel_file_path.exists():
        print("Excel file found, loading entities and relations from Excel.")
        excel_entities, excel_relations = load_excel(excel_file_path)
        print(f"Loaded {len(excel_entities)} entities from Excel.")
        print(f"Loaded {len(excel_relations)} relations from Excel.")
    else:
        print("Excel file not found.")

    if entities_json_path.exists():
        print(f"Found entities.json at: {entities_json_path}")
        entities_tmp = load_json(entities_json_path)
        json_entities = list(entities_tmp.values())
        print(f"Loaded {len(json_entities)} entities from JSON.")
    else:
        print(f"entities.json not found at: {entities_json_path}")

    if relations_json_path.exists():
        print(f"Found relations.json at: {relations_json_path}")
        relations_tmp = load_json(relations_json_path)
        json_relations = list(relations_tmp.values())
        print(f"Loaded {len(json_relations)} relations from JSON.")
    else:
        print(f"relations.json not found at: {relations_json_path}")

    # Combine entities and relations, prioritizing Excel values if available
    combined_entities = list(set(excel_entities).union(set(json_entities)))
    combined_relations = list(set(excel_relations).union(set(json_relations)))

    print(f"Total combined entities: {len(combined_entities)}")
    print(f"Total combined relations: {len(combined_relations)}")

    # Create dictionaries with assigned IDs
    entities_dict = {entity: idx for idx, entity in enumerate(combined_entities)}
    relations_dict = {relation: idx for idx, relation in enumerate(combined_relations)}

    # Check the dictionaries before writing to files
    print(f"Entities dictionary: {entities_dict}")
    print(f"Relations dictionary: {relations_dict}")

    # Define output paths
    dreeam_meta_path = Path(output_dir) / "DREEAM_META"
    dreeam_meta_path.mkdir(parents=True, exist_ok=True)

    rebel_meta_path = Path(output_dir) / "REBEL_META"
    rebel_meta_path.mkdir(parents=True, exist_ok=True)

    # Write the dictionaries to JSON files
    dreeam_entities_file = dreeam_meta_path / "df_entities.json"
    dreeam_relations_file = dreeam_meta_path / "df_rels.json"

    rebel_entities_file = rebel_meta_path / "entities.json"
    rebel_relations_file = rebel_meta_path / "relations.json"

    # Save dictionaries to JSON
    with open(dreeam_entities_file, "w") as f:
        json.dump(entities_dict, f, indent=4)
        print(f"Saved {len(entities_dict)} entities to {dreeam_entities_file}")

    with open(dreeam_relations_file, "w") as f:
        json.dump(relations_dict, f, indent=4)
        print(f"Saved {len(relations_dict)} relations to {dreeam_relations_file}")

    with open(rebel_entities_file, "w") as f:
        json.dump({k: f"<{k}>" for k, v in entities_dict.items()}, f, indent=4)
        print(f"Saved {len(entities_dict)} entities mapping to {rebel_entities_file}")

    with open(rebel_relations_file, "w") as f:
        json.dump({k: k for k, v in relations_dict.items()}, f, indent=4)
        print(f"Saved {len(relations_dict)} relations mapping to {rebel_relations_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export JSON types from Excel file")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The directory containing types.xlsx or JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory that stores all the dataset files",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
