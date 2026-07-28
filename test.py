# test.py
import os
from pathlib import Path
import argparse
import torch
from transformers import TrainingArguments, set_seed

from core.dataset import ImageFolderWithPaths, ImageClassificationCollator
from core.builders import build_model
from core.trainers import CETrainer
from evaluation.test_utils import run_test_and_save_outputs
from finetune import load_config, build_label_maps, build_image_processor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module name, e.g. swint.chu.swin_tiny_block_cbam_34_chu_full_w_5fold",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for predefined-fold datasets: reads data_root/fold_{N}/ "
             "and defaults the checkpoint to output_dir/fold_{N}/best_model",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved model dir; overrides the fold-derived default",
    )
    parser.add_argument("--test_split", type=str, default="test")

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


def load_checkpoint_state_dict(checkpoint_dir):
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)

    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
    )


def build_test_args(config):
    return TrainingArguments(
        output_dir=config["output_dir"],
        remove_unused_columns=False,
        per_device_eval_batch_size=config["eval_batch_size"],
        dataloader_num_workers=config["num_workers"],
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
    )


def main(args):
    config = load_config(args.config)
    set_seed(config["seed"])
    print_gpu_info()

    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    data_root = Path(config["data_root"])
    if args.fold is not None:
        data_root = data_root / f"fold_{args.fold}"
        config["output_dir"] = os.path.join(config["output_dir"], f"fold_{args.fold}")

    os.makedirs(config["output_dir"], exist_ok=True)

    test_dir = data_root / args.test_split
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    checkpoint_dir = args.checkpoint or os.path.join(config["output_dir"], "best_model")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"Config: {args.config}")
    print(f"Test dir: {test_dir}")
    print(f"Checkpoint: {checkpoint_dir}")

    image_processor = build_image_processor(config)

    test_dataset = ImageFolderWithPaths(
        root_dir=str(test_dir),
        image_processor=image_processor,
        image_extensions=config["image_extensions"],
    )

    class_names = test_dataset.classes
    label2id, id2label = build_label_maps(class_names)

    print(f"Classes: {class_names}")
    print(f"Test size: {len(test_dataset)}")

    # Rebuild the exact training-time architecture (this also covers the CBAM
    # models, which are plain nn.Modules and were saved as a raw state dict,
    # not via `save_pretrained`) and load the fine-tuned weights into it.
    model = build_model(config, id2label=id2label, label2id=label2id)
    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)

    collator = ImageClassificationCollator()
    test_args = build_test_args(config)

    trainer = CETrainer(
        model=model,
        args=test_args,
        data_collator=collator,
        processing_class=image_processor,
        class_weights=None,
    )

    print("\nRunning test evaluation...")
    run_test_and_save_outputs(
        trainer=trainer,
        test_dataset=test_dataset,
        idx_to_class=id2label,
        output_dir=config["output_dir"],
        config=config,
    )

    print("\nSaved test files in:", config["output_dir"])
    for filename in [
        "test_predictions.csv",
        "test_classwise_report.csv",
        "test_metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
    ]:
        print(" -", filename)


if __name__ == "__main__":
    args = get_args()
    main(args)
