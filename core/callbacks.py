import math
import json
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback
from typing import Any


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class TrainValHistoryCallback(TrainerCallback):
    def __init__(self, save_path: str):
        self.records = []
        self._latest_train_loss = None
        self._latest_lr = None
        self.save_path = save_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs and "eval_loss" not in logs and "train_loss" not in logs:
            self._latest_train_loss = logs.get("loss")
            self._latest_lr = logs.get("learning_rate")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        if "eval_loss" not in metrics:
            return

        epoch = metrics.get("epoch", state.epoch)
        epoch_int = int(math.ceil(epoch)) if epoch is not None else None

        record = {
            "epoch": epoch_int,
            "train_loss": self._latest_train_loss,
            "lr": self._latest_lr,
        }

        # save all eval_* metrics automatically
        for k, v in metrics.items():
            if k.startswith("eval_"):
                save_key = "val_" + k.replace("eval_", "")

                try:
                    record[save_key] = float(v)
                except (TypeError, ValueError):
                    record[save_key] = v

        self.records.append(record)
        save_json(self.records, self.save_path)


class PrettyLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        epoch = logs.get("epoch", state.epoch)
        epoch_int = int(math.ceil(epoch)) if epoch is not None else None

        if "loss" in logs and "eval_loss" not in logs and "train_loss" not in logs:
            print(
                f"[train-step] epoch {epoch_int} "
                f"(progress={float(epoch):.4f}) | "
                f"loss: {float(logs['loss']):.4f} | "
                f"lr: {float(logs.get('learning_rate', 0.0)):.6f}"
            )

        elif "eval_loss" in logs:
            print(
                f"[val] epoch {epoch_int} | "
                f"loss: {float(logs['eval_loss']):.4f} | "
                f"ce: {float(logs.get('eval_ce', 0.0)):.4f} | "
                f"ece: {float(logs.get('eval_ece', 0.0)):.4f} | "
                f"acc: {float(logs.get('eval_accuracy', 0.0)):.4f} | "
                f"precision_macro: {float(logs.get('eval_precision_macro', 0.0)):.4f} | "
                f"recall_macro: {float(logs.get('eval_recall_macro', 0.0)):.4f} | "
                f"f1_macro: {float(logs.get('eval_f1_macro', 0.0)):.4f} | "
                f"f1_weighted: {float(logs.get('eval_f1_weighted', 0.0)):.4f} | "
                f"auc_roc_ovr: {float(logs.get('eval_auc_roc_ovr', 0.0)):.4f} | "
                f"mAP: {float(logs.get('eval_mAP', 0.0)):.4f}"
            )
