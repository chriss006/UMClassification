# test.py
import os
from pathlib import Path
import torch
import argparse
from transformers import AutoImageProcessor, TrainingArguments, set_seed,  AutoModelForImageClassification

from configs.convnext_tiny import CONFIG
from core.dataset import ImageFolderWithPaths, ImageClassificationCollator
from core.metrics import compute_metrics
from core.trainers import FocalTrainer
from evaluation.test_utils import run_test_and_save_outputs


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--test_split", type=str, default="test_cropped")

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

    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("MPS not available → using CPU")


def build_label_maps(class_names):
    label2id = {cls_name: i for i, cls_name in enumerate(class_names)}
    id2label = {i: cls_name for cls_name, i in label2id.items()}
    return label2id, id2label


def build_test_args(config):
    use_mps = torch.backends.mps.is_available()

    return TrainingArguments(
        output_dir=config["output_dir"],
        remove_unused_columns=False,
        per_device_eval_batch_size=config["eval_batch_size"],
        dataloader_num_workers=config["num_workers"],
        fp16=False,        
        use_mps_device=use_mps,
        report_to="none",
        disable_tqdm=False,
    )


def main(args):
    set_seed(CONFIG["seed"])
    print_gpu_info()

    # Override CONFIG from CLI
    if args.data_root is not None:
        CONFIG["data_root"] = args.data_root

    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Checkpoint path
    if args.checkpoint_dir is not None:
        best_model_dir = args.checkpoint_dir
    else:
        best_model_dir = os.path.join(CONFIG["output_dir"], "best_model")

    if not os.path.exists(best_model_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {best_model_dir}")

    # Dataset paths
    data_root = Path(CONFIG["data_root"])
    train_dir = data_root / "train"
    test_dir = data_root / args.test_split

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print(f"Data root: {data_root}")
    print(f"Test split: {args.test_split}")
    print(f"Output dir: {CONFIG['output_dir']}")
    print(f"Checkpoint dir: {best_model_dir}")

    # Processor
    image_processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])

    # Datasets
    train_dataset = ImageFolderWithPaths(
        root_dir=str(train_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )

    test_dataset = ImageFolderWithPaths(
        root_dir=str(test_dir),
        image_processor=image_processor,
        image_extensions=CONFIG["image_extensions"],
    )

    class_names = train_dataset.classes
    label2id, id2label = build_label_maps(class_names)

    print(f"Classes: {class_names}")
    print(f"Test size: {len(test_dataset)}")

    # Model
    print(f"Loading model from: {best_model_dir}")

    model = AutoModelForImageClassification.from_pretrained(
        best_model_dir,
        id2label=id2label,
        label2id=label2id,
    )

    # Trainer
    collator = ImageClassificationCollator()
    test_args = build_test_args(CONFIG)

    trainer = FocalTrainer(
        model=model,
        args=test_args,
        eval_dataset=test_dataset,
        data_collator=collator,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        focal_loss_fn=None,
    )

    # Test
    print("\nRunning test evaluation...")
    run_test_and_save_outputs(
        trainer=trainer,
        test_dataset=test_dataset,
        idx_to_class=id2label,
        output_dir=CONFIG["output_dir"],
    )

    print("\nSaved test files:")
    for filename in [
        "test_predictions.csv",
        "test_classwise_report.csv",
        "test_metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
    ]:
        print(os.path.join(CONFIG["output_dir"], filename))


if __name__ == "__main__":
    args = get_args()
    main(args)