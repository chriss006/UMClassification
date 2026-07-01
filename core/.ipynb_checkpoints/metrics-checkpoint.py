import numpy as np
from typing import Dict
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,)
from sklearn.preprocessing import label_binarize
from transformers import EvalPrediction

# softmax
def logits_to_probs(logits: np.ndarray, config=None) -> np.ndarray:
    logits_t = torch.tensor(logits, dtype=torch.float32)

    probs_t = F.softmax(logits_t, dim=1)

    if (
        config is not None
        and config.get("loss_type") == "focal"
        and config.get("apply_posterior_transform", False)
    ):
        probs_t = focal_posterior_transform(
            probs_t,
            gamma=config.get("focal_gamma", 2.0),
        )

    return probs_t.cpu().numpy()

# CE
def cross_entropy(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.clip(probs, eps, 1.0)
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(true_probs)))

# Class-wise CE
def classwise_cross_entropy(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    probs = np.clip(probs, eps, 1.0)

    out = {}
    for c in range(num_classes):
        idx = labels == c

        if np.sum(idx) == 0:
            out[f"ce_class_{c}"] = float("nan")
        else:
            true_probs = probs[idx, c]
            out[f"ce_class_{c}"] = float(-np.mean(np.log(true_probs)))

    return out

# Class-wise Acc
def classwise_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    out = {}

    for c in range(num_classes):
        idx = labels == c

        if np.sum(idx) == 0:
            out[f"acc_class_{c}"] = float("nan")
        else:
            out[f"acc_class_{c}"] = float(np.mean(preds[idx] == labels[idx]))

    return out

# ECE
def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(acc_in_bin - conf_in_bin)

    return float(ece)

# Focal Posterior Transformation 
def focal_posterior_transform(prob, gamma=2.0, eps=1e-12):
    prob = torch.clamp(prob, eps, 1.0 - eps)

    numerator = prob
    denominator = (1.0 - prob) ** gamma - gamma * (1.0 - prob) ** (gamma - 1.0) * prob * torch.log(prob)

    h = numerator / torch.clamp(denominator, min=eps)

    calibrated_prob = h / torch.clamp(h.sum(dim=1, keepdim=True), min=eps)

    return calibrated_prob


def compute_metrics(eval_pred: EvalPrediction, config=None) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = logits_to_probs(logits, config=config)
    preds = np.argmax(probs, axis=1)

    num_classes = probs.shape[1]

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
        "ce": cross_entropy(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=15),
    }

    metrics.update(
        classwise_cross_entropy(
            probs=probs,
            labels=labels,
            num_classes=num_classes,
        )
    )

    metrics.update(
        classwise_accuracy(
            preds=preds,
            labels=labels,
            num_classes=num_classes,
        )
    )

    try:
        metrics["auc_roc_ovr"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auc_roc_ovr"] = float("nan")

    try:
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        ap_per_class = average_precision_score(labels_bin, probs, average=None)
        mAP = average_precision_score(labels_bin, probs, average="macro")

        metrics["mAP"] = float(mAP)

        for c, ap in enumerate(ap_per_class):
            metrics[f"ap_class_{c}"] = float(ap)

    except ValueError:
        metrics["mAP"] = float("nan")

        for c in range(num_classes):
            metrics[f"ap_class_{c}"] = float("nan")

    return metrics


def make_compute_metrics(config):
    def wrapped(eval_pred):
        return compute_metrics(eval_pred, config=config)
    return wrapped
