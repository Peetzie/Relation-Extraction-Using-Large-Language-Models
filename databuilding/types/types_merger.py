import pandas as pd

# Load the global types for reference
global_types_path = "Global types.xlsx"
global_types = pd.read_excel(global_types_path, sheet_name=None)

global_entities = global_types["Entities"]
global_relations = global_types["Relations"]

# Ensure 'Cleaned Relations' column exists in the global_relations DataFrame
if "Cleaned Relations" not in global_relations.columns:
    global_relations["Cleaned Relations"] = global_relations["Relations"]

# Ensure 'Cleaned Entities' column exists in the global_entities DataFrame
if "Cleaned Entities" not in global_entities.columns:
    global_entities["Cleaned Entities"] = global_entities["Entities"]

# Path to fils
subset_files = [
    "Conll04 types.xlsx",
    "CrossRE types.xlsx",
    "DocRED types.xlsx",
    "NYT types.xlsx",
    "SciERC types.xlsx",
]


# Function to map and combine cleaned types into the global set
def combine_with_global_types_v3(file_path, global_entities, global_relations):
    sheets = pd.read_excel(file_path, sheet_name=None)

    for sheet_name, data in sheets.items():
        if "entities" in sheet_name.lower():
            for idx, row in data.iterrows():
                entity = row["Entities"]
                cleaned_entity = row["Cleaned Entities"]
                if entity not in global_entities["Entities"].values:
                    global_entities = pd.concat(
                        [
                            global_entities,
                            pd.DataFrame(
                                {
                                    "Entities": [entity],
                                    "Cleaned Entities": [cleaned_entity],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                else:
                    global_entities.loc[
                        global_entities["Entities"] == entity, "Cleaned Entities"
                    ] = cleaned_entity
        elif "relations" in sheet_name.lower():
            for idx, row in data.iterrows():
                relation = row["Relations"]
                cleaned_relation = row["Cleaned Relations"]
                if relation not in global_relations["Relations"].values:
                    global_relations = pd.concat(
                        [
                            global_relations,
                            pd.DataFrame(
                                {
                                    "Relations": [relation],
                                    "Cleaned Relations": [cleaned_relation],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                else:
                    global_relations.loc[
                        global_relations["Relations"] == relation, "Cleaned Relations"
                    ] = cleaned_relation

    return global_entities, global_relations


# Combine the subsets with the global types
for file_path in subset_files:
    global_entities, global_relations = combine_with_global_types_v3(
        file_path, global_entities, global_relations
    )


# Save the combined global types back to the Excel file
combined_global_types_path = "Combined_Global_types.xlsx"

with pd.ExcelWriter(combined_global_types_path) as writer:
    global_entities.to_excel(writer, sheet_name="Entities", index=False)
    global_relations.to_excel(writer, sheet_name="Relations", index=False)
