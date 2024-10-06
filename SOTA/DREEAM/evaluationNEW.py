import os
import os.path
import json
import numpy as np
from tqdm import tqdm
import globals
import copy

# #rel2id = json.load(open("/work3/s174159/dreeam/meta/rel2id.json", "r"))
# #id2rel = {value: key for key, value in rel2id.items()}


# # rel2id = json.load(open('/work3/s174159/LLM_Thesis/processed_data/DocRED/meta/df_rels.json', 'r'))
# # id2rel = {value: key for key, value in rel2id.items()}


def get_title2pred(pred: list) -> dict:
    """
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    """

    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])

        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(
                    p["score"], title2pred[p["title"]][curr]
                )
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred


def get_title2gt(features: dict) -> dict:
    """
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    """
    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]):
            h, t = p
            label = np.array(f["labels"][idx])
            rs = np.nonzero(label[1:])[0] + 1  # + 1 for no-label
            title2gt[title].extend([(h, t, globals.id2rel[r]) for r in rs])

    return title2gt


def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    """
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand.
    """

    sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True)
    precs, recalls = [], []

    for pred in sorted_pred:
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall

    recalls = np.asarray(recalls, dtype="float32")
    precs = np.asarray(precs, dtype="float32")
    f1_arr = 2 * recalls * precs / (recalls + precs + 1e-20)
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print("Best thresh", thresh, "\tbest F1", f1)
    return thresh, sorted_pred[: f1_pos + 1]


def merge_results(pred: list, pred_pseudo: list, features: list, thresh: float = None):
    """
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    """

    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    title2gt = get_title2gt(features)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    titles = list(title2pred.keys())
    cand = []
    merged_res = []
    correct, num_pred = 0, 0

    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())
        for r in union:
            if r in rels and r in rels_pseudo:  # add those into predictions
                if rels[r] > 0 and rels_pseudo[r] > 0:
                    merged_res.append(
                        {"title": t, "h_idx": r[0], "t_idx": r[1], "r": r[2]}
                    )
                    num_pred += 1
                    correct += r in title2gt[t]
                    continue
                score = rels[r] + rels_pseudo[r]
            elif r in rels:  # -10 for penalty
                score = rels[r] - 10
            elif r in rels_pseudo:
                score = rels_pseudo[r] - 10
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))

    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True)
        last = min(filter(lambda x: x[1] > thresh, sorted_pred))
        until = sorted_pred.index(last)
        cand = sorted_pred[: until + 1]
        merged_res.extend(
            [{"title": r[2], "h_idx": r[3], "t_idx": r[4], "r": r[5]} for r in cand]
        )
        return merged_res, thresh

    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend(
            [{"title": r[2], "h_idx": r[3], "t_idx": r[4], "r": r[5]} for r in cand]
        )

    return merged_res, thresh


def extract_relative_score(scores: list, topks: list) -> list:
    """
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    """

    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks == 0)].item()

    scores -= na_score

    return scores


def to_official(
    preds: list,
    features: list,
    evi_preds: list = [],
    scores: list = [],
    topks: list = [],
):
    """
    Convert the predictions to official format for evaluating.
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        :scores: list of scores of topk relation labels for each entity pair.
        :topks: list of topk relation labels for each entity pair.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    """

    h_idx, t_idx, title, sents = [], [], [], []

    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sents += [len(f["sent_pos"])] * len(hts)

    official_res = []
    res = []

    for i in tqdm(range(preds.shape[0]), desc="preds"):  # for each entity pair
        if scores != []:
            score = extract_relative_score(scores[i], topks[i])
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()

        for p in pred:  # for each predicted relation label (topk)
            curr_result = {
                "title": title[i],
                "h_idx": h_idx[i],
                "t_idx": t_idx[i],
                "r": globals.id2rel[p],
            }
            if evi_preds != []:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist()
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores != []:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                official_res.append(curr_result)
            res.append(curr_result)

    return official_res, res


def gen_train_facts(data_file_name, truth_dir):

    fact_file_name = data_file_name[data_file_name.find("train_") :]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data["vertexSet"]
        for label in data["labels"]:
            rel = label["r"]
            for n1 in vertexSet[label["h"]]:
                for n2 in vertexSet[label["t"]]:
                    fact_in_train.add((n1["name"], n2["name"], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(
    tmp,
    path,
    train_file="train_annotated.json",
    dev_file="dev.json",
    out_prediction_file=None,
):
    """
    Adapted from the official evaluation code, with additional macro F1 calculation.
    """
    truth_dir = os.path.join(path, "ref")

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(
        os.path.join(path, "train_distant.json"), truth_dir
    )

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    tot_evidences = 0
    # titleset = set([])

    title2vectexSet = {}

    title2testsetrow = {}

    # print("Printing labels:")

    if out_prediction_file is not None:
        _lines = [dict() for _ in range(len(truth))]
        inc = 0

    for x in truth:
        title = x["title"]
        # titleset.add(title)

        # print("****************************")
        # print(x)
        # print(inc)
        # print("****************************")

        # if inc == 1:
        #     import time

        #     time.sleep(100000000)

        vertexSet = x["vertexSet"]
        title2vectexSet[title] = vertexSet

        # print(vertexSet)

        if "labels" not in x:  # official test set from DocRED
            continue

        for label in x["labels"]:
            # print(label)
            r = label["r"]
            h_idx = label["h"]
            t_idx = label["t"]
            # print(h_idx)
            std[(title, r, h_idx, t_idx)] = set(label["evidence"])
            tot_evidences += len(label["evidence"])
            if out_prediction_file is not None:
                _labels_line = _lines[inc].get("labels", [])
                _labels_line.append(
                    {
                        "head": vertexSet[h_idx][0]["name"],
                        "head_type": vertexSet[h_idx][0]["type"],
                        "type": r,
                        "tail": vertexSet[t_idx][0]["name"],
                        "tail_type": vertexSet[t_idx][0]["type"],
                    }
                )
                _lines[inc]["labels"] = _labels_line

        if out_prediction_file is not None:
            _lines[inc]["testFileRow"] = inc
            title2testsetrow[title] = inc
            inc += 1
    tot_relations = len(std)
    # print("Values before sorting:")
    # for item in tmp:
    #     print(
    #         f"title: {item['title']}, h_idx: {item['h_idx']}, t_idx: {item['t_idx']}, r: {item['r']}"
    #     )

    # Sort the list with enforced types in the sorting key
    tmp.sort(
        key=lambda x: (str(x["title"]), int(x["h_idx"]), int(x["t_idx"]), str(x["r"]))
    )

    submission_answer = [tmp[0]]

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x["title"], x["h_idx"], x["t_idx"], x["r"]) != (
            y["title"],
            y["h_idx"],
            y["t_idx"],
            y["r"],
        ):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])

    # Variables for macro F1 calculation
    class_precisions = {}
    class_recalls = {}
    class_f1_scores = {}
    class_counts = {}

    for x in submission_answer:
        title = x["title"]
        h_idx = x["h_idx"]
        t_idx = x["t_idx"]
        r = x["r"]
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if "evidence" in x:
            evi = set(x["evidence"])
        else:
            evi = set([])
        pred_evi += len(evi)

        if out_prediction_file is not None:
            _inc = title2testsetrow[title]
            _preds_line = _lines[_inc].get("predictions", [])
            _preds_line.append(
                {
                    "head": vertexSet[h_idx][0]["name"],
                    "head_type": vertexSet[h_idx][0]["type"],
                    "type": r,
                    "tail": vertexSet[t_idx][0]["name"],
                    "tail_type": vertexSet[t_idx][0]["type"],
                }
            )
            _lines[_inc]["predictions"] = _preds_line

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            if h_idx >= len(vertexSet) or t_idx >= len(vertexSet):  # TODO
                continue

            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1["name"], n2["name"], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1["name"], n2["name"], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

        # Update per-class precision and recall counts
        if r not in class_precisions:
            class_precisions[r] = 0
            class_recalls[r] = 0
            class_counts[r] = 0

        class_counts[r] += 1

        if (title, r, h_idx, t_idx) in std:
            class_precisions[r] += 1  # True positive for this class
            class_recalls[r] += 1  # True positive for this class

    if out_prediction_file is not None:
        for line in _lines:
            predictions = line.get("predictions", [])
            labels = line.get("labels", [])
            mod_predictions = copy.deepcopy(predictions)
            mod_labels = copy.deepcopy(labels)
            for pred in mod_predictions:
                pred["correct"] = pred in labels
            for label in mod_labels:
                label["correct"] = label in predictions
                # if label["head"] == 'Wilfried " Willi " Schneider':
                #     print(predictions)
            line["predictions"] = mod_predictions
            line["labels"] = mod_labels

        with open(out_prediction_file, "w") as f:
            f.write(json.dumps(_lines))
            f.write("\n")

    _lines = None

    # Calculate micro F1 score
    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
    re_f1 = 2.0 * re_p * re_r / (re_p + re_r) if re_p + re_r > 0 else 0

    # Calculate macro F1 score
    macro_f1_sum = 0
    n_classes = len(class_precisions)

    for r in class_precisions:
        precision = class_precisions[r] / class_counts[r] if class_counts[r] > 0 else 0
        recall = class_recalls[r] / class_counts[r] if class_counts[r] > 0 else 0
        class_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        class_f1_scores[r] = class_f1
        macro_f1_sum += class_f1

    macro_f1 = macro_f1_sum / n_classes if n_classes > 0 else 0

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0
    evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r) if evi_p + evi_r > 0 else 0

    re_p_ignore_train_annotated = (
        1.0
        * (correct_re - correct_in_train_annotated)
        / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    )
    re_p_ignore_train = (
        1.0
        * (correct_re - correct_in_train_distant)
        / (len(submission_answer) - correct_in_train_distant + 1e-5)
    )

    re_f1_ignore_train_annotated = (
        2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)
        if re_p_ignore_train_annotated + re_r > 0
        else 0
    )

    if re_p_ignore_train + re_r > 0:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)
    else:
        re_f1_ignore_train = 0

    return (
        [re_p, re_r, re_f1, macro_f1],  # Return macro F1 score as well
        [evi_p, evi_r, evi_f1, 0],
        [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated, 0],
        [re_p_ignore_train, re_r, re_f1_ignore_train, 0],
        [class_precisions, class_recalls, class_f1_scores, 0],
    )
