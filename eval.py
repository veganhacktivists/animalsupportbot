import argparse
from functools import reduce
import os
import spacy
from datetime import datetime

import pandas as pd
import numpy as np

from argmatcher import ArgMatcher
import matplotlib.pyplot as plt

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", help="Input data to be evaluated", type=str)
    parser.add_argument(
        "--n-neighbors",
        help="Number of neighbors with weighted vote",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--threshold",
        help="Similarity threshold to classify something",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--certain-threshold",
        help="Similarity threshold above which the top-1 neighbor class is chosen",
        default=0.9,
    )
    args = parser.parse_args()
    return args


def _read_csv(input_csv):
    """
    Read the input csv, converting to pandas dataframe

    Args:
        input_csv: csv file with 2 column format
            - text, argument
            - "Protein deficiency", "what_about_protein"

    Outputs:
        df: pandas dataframe
    """
    return pd.read_csv(input_csv)


def _verify_data(df):
    """
    Verify the input data, making sure fields are correct

    Args:
        df: data frame with required format
    """
    assert len(df.columns) == 2, "Found more than 2 columns"
    assert "text" in df.columns, "Didn't find 'text' column"
    assert "label" in df.columns, "Didnt' find 'label' column"


def _filter_data(df, argm):
    """
    Filters out data, removing:
        - empty text
        - labels which don't match arguments
        - text which is already in the training set

    Inputs:
        df: pd.DataFrame that contains label, text columns
        argm: ArgMatcher object

    Returns:
        reduced_df: pd.DataFrame with filtered out rows
    """
    known_args = set(list(argm.key_label_map.keys()))
    original_len = len(df)

    # Remove if label not recognized
    reduced_df = df[df.label.isin(known_args)]

    # Remove empty text field
    reduced_df = reduced_df[reduced_df.text.notnull()]

    # Remove examples in training set
    train_texts = argm.template_dict["text"]
    reduced_df = reduced_df[~reduced_df.text.isin(train_texts)]

    new_len = len(reduced_df)

    print(f"Reduced dataframe from {original_len} -> {new_len}")
    assert new_len != 0, "After filtering out bad rows, dataframe had 0 rows"

    new_num_labels_seen = len(set(reduced_df.label.values))
    num_known_args = len(known_args)

    print(f"{new_num_labels_seen} classes seen out of {num_known_args}")
    print(f"{new_num_labels_seen/num_known_args * 100}% Class coverage")
    return reduced_df


def process_eval_data(csv, argm):
    """
    Process the csv, verifying data, and returing a clean df for eval
    """
    df = _read_csv(csv)
    _verify_data(df)
    df = _filter_data(df, argm)
    texts, labels = df.text.values, df.label.values
    return texts, labels


def evaluate_model(
    argm, texts, labels, n_neighbors=3, threshold=0.0, certain_threshold=0.9
):
    """
    Evaluate the classification performance of the argmatcher
    """
    pred_labels_enc = argm.match_batch_text(
        texts,
        N_neighbors=n_neighbors,
        threshold=threshold,
        certain_threshold=certain_threshold,
    )

    # Need to encode labels
    lookup = lambda x: argm.key_label_map[x]
    true_labels_enc = np.array(list(map(lookup, labels)))

    bal_acc = balanced_accuracy_score(true_labels_enc, pred_labels_enc)
    return bal_acc, pred_labels_enc, true_labels_enc


def log_eval_results(log_folder, argm, texts, pred_labels_enc, true_labels_enc):
    """
    Log evaluation results in some folder for further inspection
    """
    ## Write the pred_labels and true_labels
    arglookup = lambda x: argm.label_key_map[x]

    raw_results_df = pd.DataFrame(
        columns=["text", "true_label_enc", "pred_label_enc", "true_label", "pred_label"]
    )
    raw_results_df["text"] = texts
    raw_results_df["true_label_enc"] = true_labels_enc
    raw_results_df["pred_label_enc"] = pred_labels_enc
    raw_results_df["true_label"] = list(map(arglookup, true_labels_enc))
    raw_results_df["pred_label"] = list(map(arglookup, pred_labels_enc))
    raw_results_df.to_csv(os.path.join(log_folder, "raw_results.csv"), index=False)

    ## Make small results output with single value metrics
    # TODO

    ## Plot confusion matrices
    # Get label ordering, removing missing args
    label_ordering = list(argm.label_key_map.keys())
    label_ordering = [l for l in label_ordering if l in set(true_labels_enc)]
    display_labels = [arglookup(l) for l in label_ordering]

    make_confusion_matrices(
        true_labels_enc, pred_labels_enc, label_ordering, display_labels, log_folder
    )


def make_confusion_matrices(y_true, y_pred, label_ordering, display_labels, outfolder):
    """
    Make confusion matrix plots
    """
    cm_pred = confusion_matrix(y_true, y_pred, labels=label_ordering, normalize="pred")
    cm_true = confusion_matrix(y_true, y_pred, labels=label_ordering, normalize="true")
    cm_none = confusion_matrix(y_true, y_pred, labels=label_ordering, normalize=None)

    cm_dict = {"true": cm_true, "pred": cm_pred, "none": cm_none}
    for k, v in cm_dict.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=v, display_labels=display_labels)
        fig, ax = plt.subplots(1, figsize=(20, 20))
        disp.plot(ax=ax, xticks_rotation="vertical", colorbar=True)
        plt.tight_layout()
        fig.savefig(os.path.join(outfolder, f"cm_{k}.png"), transparent=False)


if __name__ == "__main__":
    args = parse_args()
    eval_data = args.eval_csv

    assert os.path.isfile(eval_data), f"Couldn't locate {eval_data}"

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("universal_sentence_encoder", config={"model_name": "en_use_lg"})
    argm = ArgMatcher(nlp, None, None, preload=True)

    texts, labels = process_eval_data(eval_data, argm)

    print(
        f"Evaluting argmatcher with:\n \
        n_neighbors: {args.n_neighbors}\n \
        threshold: {args.threshold}\n \
        certain_threshold: {args.certain_threshold}"
    )

    bal_acc, pred_labels_enc, true_labels_enc = evaluate_model(
        argm,
        texts,
        labels,
        n_neighbors=args.n_neighbors,
        threshold=args.threshold,
        certain_threshold=args.certain_threshold,
    )
    print(f"Balanced accuracy is: {bal_acc}")

    os.makedirs("./eval_logs/", exist_ok=True)

    ename = os.path.splitext(os.path.basename(eval_data))[0]
    ctime = datetime.now().strftime("%H-%M-%d-%m-%Y")
    log_folder = os.path.join("./eval_logs/", f"{ename}_{ctime}")
    os.makedirs(log_folder, exist_ok=True)

    log_eval_results(log_folder, argm, texts, pred_labels_enc, true_labels_enc)
