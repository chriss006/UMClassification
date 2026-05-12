import argparse
import os
import numpy as np
import pandas as pd
import torch

from core.metrics import (
    focal_posterior_transform,
    cross_entropy,
    expected_calibration_error,
    classwise_cross_entropy,
    classwise_accuracy,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pred_filename", type=str, default="test_predictions.csv")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--num_classes", type=int, default=6)

    return parser.parse_args()


def compute_metrics_from_probs(labels, probs):
    preds = np.argmax(probs, axis=1)
    num_classes = probs.shape[1]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "ce": cross_entropy(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=15),
    }

    metrics.update(classwise_cross_entropy(probs, labels, num_classes))
    metrics.update(classwise_accuracy(preds, labels, num_classes))

    try:
        metrics["auc_roc_ovr"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auc_roc_ovr"] = float("nan")

    try:
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        ap_per_class = average_precision_score(labels_bin, probs, average=None)
        metrics["mAP"] = float(average_precision_score(labels_bin, probs, average="macro"))

        for c, ap in enumerate(ap_per_class):
            metrics[f"ap_class_{c}"] = float(ap)

    except ValueError:
        metrics["mAP"] = float("nan")
        for c in range(num_classes):
            metrics[f"ap_class_{c}"] = float("nan")

    return metrics


def process_one_fold(fold_dir, pred_filename, gamma, num_classes):
    pred_csv = os.path.join(fold_dir, pred_filename)
    output_dir = os.path.join(fold_dir, "psi")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)

    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    missing_cols = [c for c in prob_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns in {pred_csv}: {missing_cols}")

    if "true_label" in df.columns:
        label_col = "true_label"
    elif "y_true" in df.columns:
        label_col = "y_true"
    else:
        raise ValueError(f"Could not find label column in {pred_csv}")

    probs = df[prob_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy(dtype=int)

    probs_t = torch.tensor(probs, dtype=torch.float32)
    psi_probs = focal_posterior_transform(probs_t, gamma=gamma).cpu().numpy()
    psi_preds = np.argmax(psi_probs, axis=1)

    out_df = df.copy()

    for i, col in enumerate(prob_cols):
        out_df[col] = psi_probs[:, i]

    if "pred_label" in out_df.columns:
        out_df["pred_label"] = psi_preds
    elif "y_pred" in out_df.columns:
        out_df["y_pred"] = psi_preds
    else:
        out_df["pred_label"] = psi_preds

    out_df["correct"] = (psi_preds == labels).astype(int)
    out_df["confidence"] = np.max(psi_probs, axis=1)
    out_df["true_class_prob"] = psi_probs[np.arange(len(labels)), labels]

    pred_out_path = os.path.join(output_dir, "test_predictions_psi.csv")
    out_df.to_csv(pred_out_path, index=False)

    metrics = compute_metrics_from_probs(labels, psi_probs)
    metrics_out_path = os.path.join(output_dir, "test_metrics_psi.csv")
    pd.DataFrame([metrics]).to_csv(metrics_out_path, index=False)

    print(f"Saved: {pred_out_path}")
    print(f"Saved: {metrics_out_path}")

    return metrics


def main():
    args = parse_args()

    all_metrics = []

    for fold in range(args.folds):
        fold_dir = os.path.join(args.input_root, f"fold{fold}")
        print(f"\n========== Processing fold{fold} ==========")

        metrics = process_one_fold(
            fold_dir=fold_dir,
            pred_filename=args.pred_filename,
            gamma=args.gamma,
            num_classes=args.num_classes,
        )

        metrics["fold"] = fold
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[["fold"] + [c for c in metrics_df.columns if c != "fold"]]

    all_metrics_path = os.path.join(args.input_root, "psi_metrics_all_folds.csv")
    metrics_df.to_csv(all_metrics_path, index=False)

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.drop("fold")
    summary = pd.DataFrame({
        "metric": numeric_cols,
        "mean": metrics_df[numeric_cols].mean().values,
        "std": metrics_df[numeric_cols].std(ddof=1).values,
    })

    summary_path = os.path.join(args.input_root, "psi_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved all-fold metrics: {all_metrics_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()