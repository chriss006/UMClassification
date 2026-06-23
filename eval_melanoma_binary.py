"""Collapse the 3-class CHU test predictions (TMI=0, melanoma=1, naevi=2)
into melanoma vs non-melanoma binary metrics, per fold and aggregated,
for each experiment directory passed on the command line."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MELANOMA_LABEL = 1


def fold_metrics(csv_path):
    df = pd.read_csv(csv_path)
    y_true = (df["true_label"] == MELANOMA_LABEL).astype(int).values
    y_pred = (df["pred_label"] == MELANOMA_LABEL).astype(int).values
    y_score = df["prob_class_1"].values

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
    }


def evaluate_experiment(exp_dir):
    csv_paths = sorted(glob.glob(os.path.join(exp_dir, "fold*", "test_predictions.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No test_predictions.csv found under {exp_dir}")

    rows = []
    for path in csv_paths:
        fold_name = os.path.basename(os.path.dirname(path))
        m = fold_metrics(path)
        m["fold"] = fold_name
        rows.append(m)

    fold_df = pd.DataFrame(rows).set_index("fold")

    summary = fold_df.agg(["mean", "std"]).T
    summary["mean ± std"] = summary.apply(
        lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1
    )
    return fold_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="One or more experiment dirs containing fold*/test_predictions.csv")
    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        print(f"\n=== {exp_dir} (melanoma vs non-melanoma) ===")
        fold_df, summary = evaluate_experiment(exp_dir)
        print(fold_df.round(3))
        print()
        print(summary[["mean ± std"]])


if __name__ == "__main__":
    main()
