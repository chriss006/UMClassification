import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from core.metrics import (
    logits_to_probs,
    cross_entropy,
    classwise_cross_entropy,
    classwise_accuracy,
    expected_calibration_error,
)



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_test_and_save_outputs(
    trainer: Trainer,
    test_dataset: Dataset,
    idx_to_class: Dict[int, str],
    output_dir: str,
    config = None,
):

    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    probs = logits_to_probs(logits, config=config)
    preds = np.argmax(logits, axis=1)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    rows = []
    for i, sample in enumerate(test_dataset.samples):
        image_path, true_idx = sample
        pred_idx = int(preds[i])
        image_id = Path(image_path).stem

        row = {
            "image_id": image_id,
            "true_label": int(true_idx),
            "pred_label": pred_idx,
            "correct": int(pred_idx == int(true_idx)),
            "confidence": float(np.max(probs[i])),
            "true_class_prob": float(probs[i, int(true_idx)]),
        }

        for c in range(len(class_names)):
            row[f"prob_class_{c}"] = float(probs[i, c])

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    report_dict = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=6,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "label"})

    classwise_df = report_df[report_df["label"].isin(class_names)].copy()
    classwise_df.to_csv(os.path.join(output_dir, "test_classwise_report.csv"), index=False)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    num_classes = probs.shape[1]
    
    test_metrics = {
        "loss": float(pred_output.metrics.get("test_loss", float("nan"))),
        "acc": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "ce": cross_entropy(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=15),
    }

    test_metrics.update(
        classwise_cross_entropy(
            probs=probs,
            labels=labels,
            num_classes=num_classes,
        )
    )

    test_metrics.update(
        classwise_accuracy(
            preds=preds,
            labels=labels,
            num_classes=num_classes,
        )
    )

    try:
        test_metrics["auc_roc_ovr"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        test_metrics["auc_roc_ovr"] = float("nan")

    try:
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        test_metrics["mAP"] = float(
            average_precision_score(labels_bin, probs, average="macro")
        )
    except ValueError:
        test_metrics["mAP"] = float("nan")

    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    unique, counts = np.unique(preds, return_counts=True)
    pred_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    save_json(pred_dist, os.path.join(output_dir, "predicted_class_distribution.json"))

    print("Predicted distribution:", pred_dist)
    print(
        "[test] "
        f"loss: {test_metrics['loss']:.4f} | "
        f"ce: {test_metrics['ce']:.4f} | "
        f"ece: {test_metrics['ece']:.4f} | "
        f"acc: {test_metrics['acc']:.4f} | "
        f"precision_macro: {test_metrics['precision_macro']:.4f} | "
        f"recall_macro: {test_metrics['recall_macro']:.4f} | "
        f"f1_macro: {test_metrics['f1_macro']:.4f} | "
        f"f1_weighted: {test_metrics['f1_weighted']:.4f} | "
        f"auc_roc_ovr: {test_metrics['auc_roc_ovr']:.4f} | "
        f"mAP: {test_metrics['mAP']:.4f}"
    )

    return test_metrics
