# finetune.py
from pathlib import Path
import torch
import os, argparse, copy, importlib
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback

from core.dataset import ImageFolderWithPaths, ImageClassificationCollator, ImageListWithPaths
from core.builders import build_model
from core.callbacks import TrainValHistoryCallback, PrettyLogCallback
from core.losses import FocalLoss
from core.metrics import make_compute_metrics
from core.trainers import FocalTrainer
from evaluation.test_utils import run_test_and_save_outputs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_augment_transform(config):
    if not config.get("augmentation", False):
        return None
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=config.get("aug_rotation", 30)),
        T.ColorJitter(
            brightness=config.get("aug_brightness", 0.3),
            contrast=config.get("aug_contrast", 0.3),
            saturation=config.get("aug_saturation", 0.1),
        ),
        T.RandomAffine(
            degrees=0,
            translate=(config.get("aug_translate", 0.1), config.get("aug_translate", 0.1)),
            shear=config.get("aug_shear", 5),
        ),
    ])

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--kfold", type=int, default=1)
    parser.add_argument("--fold_index", type=int, default=-1)
    parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Config module name, e.g. convnext_tiny, swint_tiny"
    )
    parser.add_argument("--use_predefined_folds", action="store_true")

    return parser.parse_args()

def print_gpu_info():
    print("=== Device Info ===")

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}")

    print("torch.backends.mps.is_available():", torch.backends.mps.is_available())
    print("torch.backends.mps.is_built():", torch.backends.mps.is_built())

def load_config(config_name: str):
    module = importlib.import_module(f"configs.{config_name}")
    return copy.deepcopy(module.CONFIG)


def validate_class_consistency(train_dataset, val_dataset, test_dataset):
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError(
            f"Class mismatch detected.\n"
            f"train: {train_dataset.classes}\n"
            f"val:   {val_dataset.classes}\n"
            f"test:  {test_dataset.classes}"
        )

def freeze_backbone_except_classifier(model):
    freeze_except_keywords(model, ["classifier", "head", "fc", "score"])


def freeze_except_keywords(model, keywords):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = True

    print(f"\nTrainable parameters (keywords={keywords}):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def build_label_maps(class_names):
    label2id = {cls_name: i for i, cls_name in enumerate(class_names)}
    id2label = {i: cls_name for cls_name, i in label2id.items()}
    return label2id, id2label


def build_training_args(config):
    return TrainingArguments(
        output_dir=config["output_dir"],
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config["num_workers"],
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
        report_to="none",
        disable_tqdm=True,
    )
    
def run_one_training(config, train_dataset, val_dataset, test_dataset, image_processor, class_names):
    label2id, id2label = build_label_maps(class_names)

    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    model = build_model(config, id2label=id2label, label2id=label2id)
    if config.get("freeze_except_keywords"):
        freeze_except_keywords(model, config["freeze_except_keywords"])
    elif config.get("freeze_backbone", False):
        print("\nFreezing backbone: training classifier/head only")
        freeze_backbone_except_classifier(model)
        
    collator = ImageClassificationCollator()
    training_args = build_training_args(config)

    history_callback = TrainValHistoryCallback(
        save_path=os.path.join(config["output_dir"], "train_val_history.json")
    )

    pretty_log_callback = PrettyLogCallback()

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=config["early_stopping_threshold"],
    )

    loss_type = config.get("loss_type", "focal")

    if loss_type == "focal":
        focal_loss_fn = FocalLoss(
            gamma=config["focal_gamma"],
            alpha=config["focal_alpha"],
        )
    
        trainer = FocalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=image_processor,
            compute_metrics=make_compute_metrics(config),
            callbacks=[
                history_callback,
                pretty_log_callback,
                early_stopping_callback,
            ],
            focal_loss_fn=focal_loss_fn,
        )
    
    elif loss_type == "ce":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=image_processor,
            compute_metrics=make_compute_metrics(config),
            callbacks=[
                history_callback,
                pretty_log_callback,
                early_stopping_callback,
            ],
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    trainer.remove_callback(PrinterCallback)

    print("[Train]")
    trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])

    print("\nSaving best model...")
    trainer.save_model(os.path.join(config["output_dir"], "best_model"))

    print("[Test Eval]")
    run_test_and_save_outputs(
        trainer=trainer,
        test_dataset=test_dataset,
        idx_to_class=id2label,
        output_dir=config["output_dir"],
        config=config
    )

def main(args):
    CONFIG = load_config(args.config)
    set_seed(CONFIG["seed"])

    if args.data_root is not None:
        CONFIG["data_root"] = args.data_root

    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir

    ensure_dir(CONFIG["output_dir"])
    print_gpu_info()

    data_root = Path(CONFIG["data_root"])
    original_output_dir = CONFIG["output_dir"]

    image_processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])
    augment_transform = build_augment_transform(CONFIG)


    if args.use_predefined_folds:
        fold_dirs = sorted([
            p for p in data_root.iterdir()
            if p.is_dir() and p.name.startswith("fold_")
        ])

        if len(fold_dirs) == 0:
            raise ValueError(f"No predefined fold folders found in: {data_root}")

        for fold_dir in fold_dirs:
            print(f"\n========== Running predefined {fold_dir.name} ==========")

            train_dir = fold_dir / "train"
            val_dir = fold_dir / "val"
            test_dir = fold_dir / "test"

            if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
                raise FileNotFoundError(
                    f"{fold_dir} must contain train/, val/, and test/ folders."
                )

            train_dataset = ImageFolderWithPaths(
                root_dir=str(train_dir),
                image_processor=image_processor,
                image_extensions=CONFIG["image_extensions"],
                augment_transform=augment_transform,
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

            fold_config = copy.deepcopy(CONFIG)
            fold_config["data_root"] = str(fold_dir)
            fold_config["output_dir"] = os.path.join(original_output_dir, fold_dir.name)
            ensure_dir(fold_config["output_dir"])

            fold_num = fold_dir.name.replace("fold_", "")
            if "pretrained_checkpoint" in fold_config:
                fold_config["pretrained_checkpoint"] = fold_config["pretrained_checkpoint"].format(
                    fold=fold_num
                )
            if fold_config.get("cbam_init_checkpoint"):
                fold_config["cbam_init_checkpoint"] = fold_config["cbam_init_checkpoint"].format(
                    fold=fold_num
                )

            run_one_training(
                config=fold_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                image_processor=image_processor,
                class_names=class_names,
            )

        return


    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_dataset_raw = ImageFolderWithPaths(
        root_dir=str(train_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
        augment_transform=augment_transform,
    )

    val_dataset_raw = ImageFolderWithPaths(
        root_dir=str(val_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )

    test_dataset = ImageFolderWithPaths(
        root_dir=str(test_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )

    validate_class_consistency(train_dataset_raw, val_dataset_raw, test_dataset)

    class_names = train_dataset_raw.classes


    if args.kfold > 1:
        all_samples = train_dataset_raw.samples + val_dataset_raw.samples
        labels = np.array([label for _, label in all_samples])

        skf = StratifiedKFold(
            n_splits=args.kfold,
            shuffle=True,
            random_state=CONFIG["seed"],
        )

        folds = list(skf.split(np.zeros(len(labels)), labels))
        fold_indices = [args.fold_index] if args.fold_index >= 0 else list(range(args.kfold))

        for fold in fold_indices:
            print(f"\n========== Running fold {fold}/{args.kfold - 1} ==========")

            train_idx, val_idx = folds[fold]

            train_samples = [all_samples[i] for i in train_idx]
            val_samples = [all_samples[i] for i in val_idx]

            train_dataset = ImageListWithPaths(
                samples=train_samples,
                image_processor=image_processor,
                classes=class_names,
                augment_transform=augment_transform,
            )

            val_dataset = ImageListWithPaths(
                samples=val_samples,
                image_processor=image_processor,
                classes=class_names,
            )

            fold_config = copy.deepcopy(CONFIG)
            fold_config["output_dir"] = os.path.join(original_output_dir, f"fold{fold}")
            ensure_dir(fold_config["output_dir"])

            if "pretrained_checkpoint" in fold_config:
                fold_config["pretrained_checkpoint"] = fold_config["pretrained_checkpoint"].format(fold=fold)
            if fold_config.get("cbam_init_checkpoint"):
                fold_config["cbam_init_checkpoint"] = fold_config["cbam_init_checkpoint"].format(fold=fold)

            run_one_training(
                config=fold_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                image_processor=image_processor,
                class_names=class_names,
            )

    else:
        run_one_training(
            config=CONFIG,
            train_dataset=train_dataset_raw,
            val_dataset=val_dataset_raw,
            test_dataset=test_dataset,
            image_processor=image_processor,
            class_names=class_names,
        )
        
if __name__ == "__main__":
    args = get_args()
    main(args)
