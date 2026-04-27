import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,)
from sklearn.preprocessing import label_binarize
from transformers import EvalPrediction


def softmax_numpy(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = softmax_numpy(logits, axis=1)
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    try:
        auc_roc_ovr = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        metrics["auc_roc_ovr"] = float(auc_roc_ovr)
    except ValueError:
        metrics["auc_roc_ovr"] = float("nan")

    try:
        num_classes = probs.shape[1]
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        ap_per_class = average_precision_score(labels_bin, probs, average=None)
        mAP = average_precision_score(labels_bin, probs, average="macro")

        metrics["mAP"] = float(mAP)
        for c, ap in enumerate(ap_per_class):
            metrics[f"ap_class_{c}"] = float(ap)
    except ValueError:
        metrics["mAP"] = float("nan")
        for c in range(probs.shape[1]):
            metrics[f"ap_class_{c}"] = float("nan")

    return metrics
