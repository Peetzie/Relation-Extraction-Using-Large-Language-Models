# %%
import dspy
import nest_asyncio
from pydantic import BaseModel, Field
from dspy.functional import TypedPredictor
from IPython.display import Markdown, display
from typing import List, Optional, Union
from dotenv import load_dotenv
from devtools import pprint
import os

assert load_dotenv("/work3/s174159/LLM_Thesis/prompt/.env") == True

import pandas as pd

from token_count import TokenCount
from tqdm import tqdm
import math

# %%
# Prepare the DSPY dataset

df = pd.read_json(
    "/work3/s174159/LLM_Thesis/data/No_Constraints/Combined/combined_dataset_corrected.json"
)
df2 = pd.read_json(
    "/work3/s174159/LLM_Thesis/data/No_Constraints/DocRED_Distant/combined_dataset_corrected.json"
)

frames = [df, df2]
df = pd.concat(frames)
dataset = []
for title, original_file_path, sents, vertexSet, labels, org_dataset, domains in tqdm(
    df.values, desc="Processing examples", total=len(df)
):
    # Convert JSON strings to Python objects
    sents = sents
    vertexSet = vertexSet
    labels = labels

    # Extract entity types and relation types for the current row
    entity_types = [
        entity["type"] for entity_list in vertexSet for entity in entity_list
    ]
    relation_types = [relation["r"] for relation in labels]

    # Create a DSPY example and append to the dataset
    example = dspy.Example(
        sentences=str(sents), entities=vertexSet, relations=labels
    ).with_inputs("sentences", "entity_types", "relation_types")

    # Attach entity_types and relation_types to the example
    example["entity_types"] = str(entity_types)
    example["relation_types"] = str(relation_types)

    dataset.append(example)

# Now the dataset is ready for use with DSPY
print(f"Total examples created: {len(dataset)}")

# Check one example to see the structure
print(dataset[0])


# %%
with open("/work3/s174159/LLM_Thesis/prompt/texted_text.txt", "w") as f:
    for line in dataset:
        f.write(str(line) + "\n")


# %%
def analyze_text_file(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()

            char_count = len(content)
            word_count = len(content.split())
            space_count = content.count(" ")
            line_count = content.count("\n")

            print("File analysis summary:")
            print("Character count:", char_count)
            print("Word count:", word_count)
            print("Space count:", space_count)
            print("Line count:", line_count)

    except FileNotFoundError:
        print("File not found!")


# %%
fp = "/work3/s174159/LLM_Thesis/prompt/texted_text.txt"

# %%
analyze_text_file(fp)

# %% [markdown]
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
#
#
# Based on this source we get an estimate

# %%
chars = 678036080
approx_tokens_chars = 4

# %%
est_tokens = math.ceil(chars / approx_tokens_chars)
print(est_tokens)

# %%
price_per_token = 5.0 / 1000000

# %%
print(f"Estimated cost: {est_tokens * price_per_token} USD")

# %% [markdown]
# ## Cheap alternative

# %%
price_per_token = 0.150 / 1000000

# %%
print(f"Estimated cost: {est_tokens * price_per_token} USD")

# %%
# delete the text file again
os.remove(fp)

# %%
