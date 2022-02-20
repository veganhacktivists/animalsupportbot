import argparse
from functools import reduce
import os
import spacy

import pandas as pd
import numpy as np

from argmatcher import ArgMatcher

from sklearn.metrics import balanced_accuracy_score


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
    Filters out empty text rows, along with
    labels which doesn't match existing arguments
    """
    known_args = set(list(argm.key_label_map.keys()))
    original_len = len(df)

    reduced_df = df[df.label.isin(known_args)]
    reduced_df = reduced_df[reduced_df.text.notnull()]

    new_len = len(reduced_df)

    print(f"Reduced dataframe from {original_len} -> {new_len}")
    assert new_len != 0, "After filtering out illegal rows, dataframe had 0 rows"

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
    pred_labels = argm.match_batch_text(
        texts,
        N_neighbors=n_neighbors,
        threshold=threshold,
        certain_threshold=certain_threshold,
    )

    # Need to encode labels
    lookup = lambda x: argm.key_label_map[x]
    encoded_labels = np.array(list(map(lookup, labels)))

    bal_acc = balanced_accuracy_score(encoded_labels, pred_labels)
    return bal_acc, pred_labels, encoded_labels


def log_eval_results(log_folder, argm, bal_acc, pred_labels, encoded_labels):
    """
    Log evaluation results in some folder for further inspection
    """
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    eval_data = args.eval_csv

    assert os.path.isfile(eval_data), f"Couldn't locate {eval_data}"

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("universal_sentence_encoder", config={"model_name": "en_use_lg"})
    argm = ArgMatcher(nlp, None, None, preload=True)

    texts, labels = process_eval_data(eval_data, argm)

    print(
        f"Evaluting argmatcher with:\n\t \
        n_neighbors: {args.n_neighbors}\n\t \
        threshold: {args.threshold}\n\t \
        certain_threshold: {args.certain_threshold}"
    )

    bal_acc, pred_labels, encoded_labels = evaluate_model(
        argm,
        texts,
        labels,
        n_neighbors=args.n_neighbors,
        threshold=args.threshold,
        certain_threshold=args.certain_threshold,
    )
    print(f"Balanced accuracy is: {bal_acc}")

    os.makedirs("./eval_logs/", exist_ok=True)
