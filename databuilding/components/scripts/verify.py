import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import argparse
from colorama import Fore, Style, init
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from download_utils import *


def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    """Add entity marker (*) at the end and beginning of entities."""
    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample["sents"]):
        # Add * marks to the beginning and end of entities
        new_map = {}
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end))
        sent_start = sent_end

        # Update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos


def integrity_check(
    json_file, tokenizer, docred_rel2id, max_seq_length, constraint=False
):
    pos_samples = 0
    neg_samples = 0
    rows_to_remove = []
    data = read_json_with_progress_no_lines(json_file)
    # with open(json_file, "r") as fh:
    #     print(json_file)
    #     data = json.load(fh)
    for doc_id in tqdm(range(len(data)), desc="Loading examples"):
        sample = data[doc_id]
        entities = sample["vertexSet"]
        entity_start, entity_end = [], []
        # Record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0]))
                entity_end.append((sent_id, pos[1] - 1))

        # Add entity markers
        sents, sent_map, sent_pos = add_entity_markers(
            sample, tokenizer, entity_start, entity_end
        )

        # Training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in sample:
            for label in sample["labels"]:
                evidence = label["evidence"]
                r = int(docred_rel2id[label["r"]])

                # Update training triples
                if (label["h"], label["t"]) not in train_triple:
                    train_triple[(label["h"], label["t"])] = [
                        {"relation": r, "evidence": evidence}
                    ]
                else:
                    train_triple[(label["h"], label["t"])].append(
                        {"relation": r, "evidence": evidence}
                    )

        # Entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end))

        relations, hts, sent_labels = [], [], []

        for h, t in train_triple.keys():  # For every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)

            for mention in train_triple[
                h, t
            ]:  # For each relation mention with head h and tail t
                relation[mention["relation"]] = 1
                for idx in mention["evidence"]:
                    sent_evi[idx] += 1

            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                # All entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts:  # And [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)
                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1
        if constraint:
            if not len(relations) == len(entities) * (len(entities) - 1):
                rows_to_remove.append(doc_id)
                print(
                    f"Row {doc_id} has {len(relations)} relations but should have {len(entities) * (len(entities) - 1)}"
                )
                print(relations)
        if not len(sents) < max_seq_length:
            rows_to_remove.append(doc_id)
            print(
                f"Row {doc_id} has {len(sents)} tokens but should have less than {max_seq_length}"
            )

    indexes_to_remove = list(set(rows_to_remove))
    if len(indexes_to_remove) > 0:
        print(indexes_to_remove[0])
        print(data[indexes_to_remove[0]])
        print(Fore.RED + "Rows to be removed: ", indexes_to_remove)
        print(Fore.RED + "total: ", len(indexes_to_remove))
        print(Style.RESET_ALL)
        # Sort indexes in reverse order to avoid index shift issues
        indexes_to_remove.sort(reverse=True)
        # Remove items by index
        for index in tqdm(
            indexes_to_remove, desc="Removing rows", total=len(indexes_to_remove)
        ):
            del data[index]
        with open(json_file, "w") as fh:
            json.dump(data, fh, indent=4)
        print("Rechecking integrity")
        integrity_check(json_file, tokenizer, docred_rel2id, max_seq_length)
    if len(indexes_to_remove) == 0:
        print(Fore.GREEN + "All rows are correct")
        print(Style.RESET_ALL)


def main(input_file, output_dir, model_name, max_seq_length, constraint=False):
    # root_dir = str(
    #     Path(input_file).parents[2]
    # )  # Assumes input_file is in the processed_data_dir
    # print(root_dir)
    root_dir = Path(root_folder, "tmp")
    TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, cache_dir=Path(root_dir, "cache_dir")
    )
    # combined_df = pd.DataFrame(
    #     read_json_with_progress_no_lines(input_file, orient="records", lines=False)
    # )

    META_FOLDER = Path(output_dir, "DREEAM_META")
    docred_rel2id = json.load(open(Path(META_FOLDER, "df_rels.json"), "r"))
    if constraint:
        integrity_check(
            input_file, TOKENIZER, docred_rel2id, max_seq_length, constraint
        )
    else:
        integrity_check(
            input_file, TOKENIZER, docred_rel2id, max_seq_length, constraint=False
        )
    print("Verification completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify and clean the combined dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The path to the combined dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory that stores all the dataset files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="The name of the pre-trained model to use",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="The maximum sequence length for the tokenizer",
    )
    parser.add_argument(
        "--constraints",
        action="store_true",
        help="Apply constraints (Relation numbers vs entities)",
    )
    args = parser.parse_args()
    if args.constraints:
        main(
            args.input_file,
            args.output_dir,
            args.model_name,
            args.max_seq_length,
            args.constraints,
        )
    else:
        main(args.input_file, args.output_dir, args.model_name, args.max_seq_length)
