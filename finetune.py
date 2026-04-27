# train.py

import os
from pathlib import Path
import torch

from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

from transformers.trainer_callback import PrinterCallback

from configs.convnext_tiny import CONFIG
from core.dataset import ImageFolderWithPaths, ImageClassificationCollator
from core.builders import build_model
from core.callbacks import TrainValHistoryCallback, PrettyLogCallback
from core.losses import FocalLoss
from core.metrics import compute_metrics
from core.trainers import FocalTrainer
from evaluation.test_utils import run_test_and_save_outputs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def print_gpu_info():
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def validate_class_consistency(train_dataset, val_dataset, test_dataset):
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError(
            f"Class mismatch detected.\n"
            f"train: {train_dataset.classes}\n"
            f"val:   {val_dataset.classes}\n"
            f"test:  {test_dataset.classes}"
        )


def build_label_maps(class_names):
    label2id = {cls_name: i for i, cls_name in enumerate(class_names)}
    id2label = {i: cls_name for cls_name, i in label2id.items()}
    return label2id, id2label


def build_training_args(config):
    return TrainingArguments(
        output_dir=config["output_dir"],
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        dataloader_num_workers=config["num_workers"],
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
        report_to="none",
        disable_tqdm=True,
    )


def main():
    set_seed(CONFIG["seed"])
    ensure_dir(CONFIG["output_dir"])
    print_gpu_info()

    data_root = Path(CONFIG["data_root"])
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    image_processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])

    train_dataset = ImageFolderWithPaths(
        root_dir=str(train_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )
    val_dataset = ImageFolderWithPaths(
        root_dir=str(val_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )
    test_dataset = ImageFolderWithPaths(
        root_dir=str(test_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )

    validate_class_consistency(train_dataset, val_dataset, test_dataset)

    class_names = train_dataset.classes
    label2id, id2label = build_label_maps(class_names)

    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    model = build_model(CONFIG, id2label=id2label, label2id=label2id)
    collator = ImageClassificationCollator()
    training_args = build_training_args(CONFIG)

    history_callback = TrainValHistoryCallback(
        save_path=os.path.join(CONFIG["output_dir"], "train_val_history.json")
    )

    pretty_log_callback = PrettyLogCallback()

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["early_stopping_patience"],
        early_stopping_threshold=CONFIG["early_stopping_threshold"],
    )

    focal_loss_fn = FocalLoss(
        gamma=CONFIG["focal_gamma"],
        alpha=CONFIG["focal_alpha"],
    )

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        callbacks=[
            history_callback,
            pretty_log_callback,
            early_stopping_callback,
        ],
        focal_loss_fn=focal_loss_fn,
    )

    trainer.remove_callback(PrinterCallback)

    print("\nStart training...")
    trainer.train(resume_from_checkpoint=CONFIG["resume_from_checkpoint"])

    print("\nSaving best model...")
    trainer.save_model(os.path.join(CONFIG["output_dir"], "best_model"))

    print("\nRunning test evaluation...")
    run_test_and_save_outputs(
        trainer=trainer,
        test_dataset=test_dataset,
        idx_to_class=id2label,
        output_dir=CONFIG["output_dir"],
    )

    print("\nSaved files:")
    for filename in [
        "best_model",
        "train_val_history.json",
        "test_predictions.csv",
        "test_classwise_report.csv",
        "test_metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
    ]:
        print(os.path.join(CONFIG["output_dir"], filename))


if __name__ == "__main__":
    main()