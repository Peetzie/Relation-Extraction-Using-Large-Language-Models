import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
from pathlib import Path

def handle_rare_classes(df, col, min_count=2):
    value_counts = df[col].value_counts()
    rare_classes = value_counts[value_counts < min_count].index
    df[col] = df[col].apply(lambda x: 'other' if x in rare_classes else x)
    return df

def stratified_split(df, stratify_col, test_size=0.2, random_state=42):
    value_counts = df[stratify_col].value_counts()
    if value_counts.min() < 2:
        rare_classes = value_counts[value_counts < 2].index
        df[stratify_col] = df[stratify_col].apply(lambda x: 'other' if x in rare_classes else x)
    
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])
    return train, test

def split_dataset(df, entity_col='vertexSet', relation_col='labels', test_size=0.2, dev_size=0.2, random_state=42):
    df['entity_stratify_col'] = df[entity_col].apply(lambda x: str(sorted(Counter([e['type'] for v in x for e in v]).items())))
    df['relation_stratify_col'] = df[relation_col].apply(lambda x: str(sorted(Counter([r['r'] for r in x]).items())))

    # Handle rare classes
    df = handle_rare_classes(df, 'entity_stratify_col')
    df = handle_rare_classes(df, 'relation_stratify_col')

    # Perform stratified splits
    train, remainder = stratified_split(df, 'entity_stratify_col', test_size=(test_size + dev_size), random_state=random_state)
    dev, test = stratified_split(remainder, 'entity_stratify_col', test_size=(test_size / (test_size + dev_size)), random_state=random_state)

    train = train.drop(columns=['entity_stratify_col', 'relation_stratify_col'])
    dev = dev.drop(columns=['entity_stratify_col', 'relation_stratify_col'])
    test = test.drop(columns=['entity_stratify_col', 'relation_stratify_col'])

    return train, dev, test

def create_split_with_distribution(df, use_original_split=True, test_size=0.2, dev_size=0.2, random_state=42):
    if use_original_split:
        # Original split logic
        distant = df[df["original_file_path"].str.contains("distant")]
        un_distant = df[~df["original_file_path"].str.contains("distant")]
        test = un_distant[un_distant["original_file_path"].str.contains("test")]
        remainder = un_distant[~un_distant["original_file_path"].str.contains("test")]
        
        if not gen_new_distant:
            remainder = pd.concat([distant, remainder])
        
        train, dev = train_test_split(remainder, test_size=0.5, random_state=random_state)
        
        return train, dev, test
    else:
        return split_dataset(df, test_size=test_size, dev_size=dev_size, random_state=random_state)

def plot_comparison_distribution(train_stratified, dev_stratified, test_stratified, train_random, dev_random, test_random, title, entity_or_relation='entity'):
    if entity_or_relation == 'entity':
        fig, axs = plt.subplots(3, 1, figsize=(12, 20))  # Adjusted height for entities
    else:
        fig, axs = plt.subplots(3, 1, figsize=(12, 50))  # Adjusted height for relations

    splits = ['Train', 'Dev', 'Test']
    for i, split in enumerate([train_stratified, dev_stratified, test_stratified]):
        if entity_or_relation == 'entity':
            entities_stratified = [v['type'] for row in split.itertuples() for ver in row.vertexSet for v in ver]
            entities_random = [v['type'] for row in [train_random, dev_random, test_random][i].itertuples() for ver in row.vertexSet for v in ver]
        else:
            entities_stratified = [r['r'] for row in split.itertuples() for r in row.labels]
            entities_random = [r['r'] for row in [train_random, dev_random, test_random][i].itertuples() for r in row.labels]

        # Debugging: Print lengths to ensure data is being processed
        print(f"{splits[i]} Set - {entity_or_relation.capitalize()} Stratified: {len(entities_stratified)}")
        print(f"{splits[i]} Set - {entity_or_relation.capitalize()} Random: {len(entities_random)}")
        
        counter_stratified = Counter(entities_stratified)
        counter_random = Counter(entities_random)
        
        labels = list(set(counter_stratified.keys()).union(set(counter_random.keys())))
        stratified_counts = [counter_stratified.get(label, 0) for label in labels]
        random_counts = [counter_random.get(label, 0) for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axs[i].barh(x - width/2, stratified_counts, width, label='Stratified', color='skyblue')
        axs[i].barh(x + width/2, random_counts, width, label='Random', color='salmon')
        axs[i].set_yticks(x)
        axs[i].set_yticklabels(labels)
        axs[i].invert_yaxis()
        axs[i].set_xlabel('Count')
        axs[i].set_title(f'{splits[i]} Set - {title}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def find_missing_types(train, dev, test):
    train_entities = set(v['type'] for row in train.itertuples() for ver in row.vertexSet for v in ver)
    dev_entities = set(v['type'] for row in dev.itertuples() for ver in row.vertexSet for v in ver)
    test_entities = set(v['type'] for row in test.itertuples() for ver in row.vertexSet for v in ver)

    train_relations = set(r['r'] for row in train.itertuples() for r in row.labels)
    dev_relations = set(r['r'] for row in dev.itertuples() for r in row.labels)
    test_relations = set(r['r'] for row in test.itertuples() for r in row.labels)

    missing_entities = {
        "in_dev_not_train": dev_entities - train_entities,
        "in_test_not_train": test_entities - train_entities,
        "in_train_not_dev": train_entities - dev_entities,
        "in_train_not_test": train_entities - test_entities
    }

    missing_relations = {
        "in_dev_not_train": dev_relations - train_relations,
        "in_test_not_train": test_relations - train_relations,
        "in_train_not_dev": train_relations - dev_relations,
        "in_train_not_test": train_relations - test_relations
    }

    return missing_entities, missing_relations

# Load the dataset
root_dir = "/Users/peetz/Documents/GitHub/LLM_Thesis"
data_dir = Path(root_dir, "data")
DocRED = Path(data_dir, "pre-processed data", "DocRED","combined_dataset.json")
df = pd.read_json(DocRED, orient="records", lines=False)

# Generate the splits with stratification
train_stratified, dev_stratified, test_stratified = create_split_with_distribution(df, use_original_split=False, test_size=0.2, dev_size=0.2, random_state=42)

# Generate the splits without stratification (original method)
gen_new_distant = False  # Assuming this flag is false based on provided code
train_random, dev_random, test_random = create_split_with_distribution(df, use_original_split=True, test_size=0.2, dev_size=0.2, random_state=42)

# Plot distributions with stratification vs without stratification
print("Comparison of Distributions:")
plot_comparison_distribution(train_stratified, dev_stratified, test_stratified, train_random, dev_random, test_random, "Entity Distribution", entity_or_relation='entity')
plot_comparison_distribution(train_stratified, dev_stratified, test_stratified, train_random, dev_random, test_random, "Relation Distribution", entity_or_relation='relation')

# Find missing entity and relation types in train set for stratified splits
missing_entities_stratified, missing_relations_stratified = find_missing_types(train_stratified, dev_stratified, test_stratified)
print(f"Entity types missing in stratified splits:\n{missing_entities_stratified}")
print(f"Relation types missing in stratified splits:\n{missing_relations_stratified}")

# Find missing entity and relation types in train set for random splits
missing_entities_random, missing_relations_random = find_missing_types(train_random, dev_random, test_random)
print(f"Entity types missing in random splits:\n{missing_entities_random}")
print(f"Relation types missing in random splits:\n{missing_relations_random}")
