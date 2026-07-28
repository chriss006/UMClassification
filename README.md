# Comparing Deep Architectures for Uveal Melanoma Classification in Ultra-Widefield Fundus Images

> **MICCAI 2026 Workshop OMIA** · Submission #36 · CC BY 4.0

**Authors:** Haehyun Lee, Sacha Nahon-Esteve, Diane Lingrand, Frederic Precioso, Pierre-Alexandre Mattei, Celia Maschi, Stéphanie Baillif

---

## Overview

We systematically compare three deep learning architectures for uveal melanoma (UM) classification from single-modality ultra-widefield (UWF) color fundus images:

- **RETFound** (ViT-L/16, CFP pre-trained)
- **EfficientNetV2-S + CBAM** 
- **Swin Transformer Tiny + Block-Level CBAM** 

Two classification tasks are evaluated on a private clinical CHU dataset:
- **3-class**: Uveal Melanoma (UM) vs. IMT vs. Naevi
- **Binary**: Melanoma vs. Other (IMT + Naevi)

All experiments use **5-fold cross-validation** with patient-level splits.

---

## Architecture: Block-Level CBAM for Swin Transformer

CAM and SAM are injected **inside each Swin Transformer block** as `forward_pre_hook`s on the attention module, after `LayerNorm` and before the attention operation:

- **Even-indexed blocks** (W-MSA): **CAM** (Channel Attention) is applied before W-MSA
- **Odd-indexed blocks** (SW-MSA): **SAM** (Spatial Attention) is applied before SW-MSA

```
W-MSA block:   LayerNorm → [CAM] → W-MSA  → residual → LayerNorm → FFN → residual
SW-MSA block:  LayerNorm → [SAM] → SW-MSA → residual → LayerNorm → FFN → residual
```

The key ablation is **at which Swin stages** this block-level injection is applied:

| Variant | Stage indices | Trainable | Matching `configs/swint/chu/` prefix |
|---|---|---|---|
| Stage 4 (frozen backbone) | `[3]` | stage-4 blocks, CBAM, head only | `swin_tiny_block_cbam_chu_5fold` |
| Stage 4 (full fine-tuning) | `[3]` | everything | `swin_tiny_block_cbam_chu_full_5fold` |
| Stage 3 & 4 | `[2, 3]` | everything | `swin_tiny_block_cbam_34_chu_full_5fold` |
| All Stages | `[0, 1, 2, 3]` | everything | `swin_tiny_block_cbam_all_chu_full_5fold` |

Each variant has a plain CE version and a `_w_5fold` (weighted CE, inverse-class-frequency) version, plus a `_5fold_binary` counterpart for the binary task. `swin_tiny_chu_full_5fold` is the CBAM-free SwinT baseline.

---

## Repository Structure

```
UMClassification/
├── finetune.py                  # Train + automatically evaluate on the test split
├── test.py                      # Standalone re-evaluation of an already-trained checkpoint
├── eval_melanoma_binary.py      # Collapse 3-class fold predictions into melanoma-vs-other metrics
├── psi_transform.py             # Post-hoc posterior calibration (focal Ψ-transform) on saved predictions
├── core/
│   ├── cbam.py                  # CBAM + SwinWithBlockCBAM / SwinWithStageCBAM / EfficientNetV2WithCBAM
│   ├── builders.py               # Model factory (SwinT, EfficientNetV2, ConvNext, RETFound, ± CBAM)
│   ├── dataset.py                # ImageFolderWithPaths, collators
│   ├── losses.py                 # FocalLoss
│   ├── metrics.py                # Accuracy, macro F1, AUC-ROC, mAP, calibration
│   ├── trainers.py               # FocalTrainer / CETrainer (custom HuggingFace Trainers)
│   └── callbacks.py              # TrainValHistoryCallback, PrettyLogCallback
├── configs/
│   ├── swint/chu/                # SwinT Block-CBAM configs actually used for the paper (5-fold, CHU data)
│   ├── swint/uwf/                # SwinT UWF pre-training configs 
│   ├── efficientnet/             # EfficientNetV2 + single CBAM configs
│   └── convnext/                 # ConvNext configs (ablation, not in the paper's main comparison)
└── evaluation/
    └── test_utils.py             # run_test_and_save_outputs()
```

---

## Requirements

```bash
python >= 3.10
torch >= 2.0
transformers >= 4.40
torchvision
timm
scikit-learn
pandas
numpy
tqdm
albumentations
```

---

## Dataset

The CHU dataset is a **private clinical dataset** from CHU de Nice (France) and is **not publicly available**.

To use your own dataset, organize images as:
```
dataset/
└── fold_{0..4}/
    ├── train/{class_name}/*.jpg
    ├── val/{class_name}/*.jpg
    └── test/{class_name}/*.jpg
```

Update `data_root` in the config file to point at the `dataset/` folder (the one containing `fold_0` … `fold_4`).

Class labels are assigned alphabetically by folder name. For the 3-class task: `imt=0, melanoma=1, naevi=2`.

---

## Training

`finetune.py` supports two ways of splitting data; the paper's 5-fold results all use the first one.

### Predefined 5-fold split

`--use_predefined_folds` 
  - iterates over every `fold_*` directory under `data_root` in a single run, writing each fold's outputs to `output_dir/fold_0/`, `output_dir/fold_1/`, …

```bash
# Stage 3 & 4, Weighted CE, 3-class, all 5 folds
python finetune.py \
  --config swint.chu.swin_tiny_block_cbam_34_chu_full_w_5fold \
  --use_predefined_folds
```

```bash
# Stage 3 & 4, Weighted CE, binary
python finetune.py \
  --config swint.chu.swin_tiny_block_cbam_34_chu_full_w_5fold_binary \
  --use_predefined_folds
```

`--config` is a **dotted module path** under `configs/` (no `.py`, no leading `configs.`), because `finetune.py` loads it with `importlib.import_module(f"configs.{args.config}")`.

### K-fold (single `train/` + `val/` folder, split internally with `StratifiedKFold`)

```bash
python finetune.py --config convnext.convnext_tiny_cbam --kfold 5 --fold_index 0
```

Omit `--fold_index` to run all `--kfold` folds sequentially in one call.

### Key config parameters

| Parameter | Description |
|---|---|
| `cbam_stage_indices` | Swin stage indices to insert CBAM (0-indexed: 0–3) |
| `cbam_mode` | `"block"` (per-block) or `"stage"` (post-stage) |
| `loss_type` | `"ce"` (cross-entropy) or `"focal"` (default; needs `focal_gamma`/`focal_alpha`) |
| `class_weights` | `None` (default), `"balanced"`, `"balanced_sqrt"`, or an explicit list — only used when `loss_type="ce"`. This is what the `_w_5fold` (Weighted CE) configs set. |
| `freeze_except_keywords` | List of parameter-name substrings to keep trainable; everything else is frozen (used for the "Stage 4, frozen backbone" variant) |
| `pretrained_checkpoint` | UWF pre-trained checkpoint path (`{fold}` is substituted per fold) |
| `num_train_epochs` | Default: 50 |
| `early_stopping_patience` | Default: 10 |

---

## Evaluation

**Evaluation is not a separate step for normal training runs.** At the end of every fold, `finetune.py` saves the best checkpoint and immediately calls `run_test_and_save_outputs()` on that fold's `test/` split ([finetune.py](finetune.py) → `run_one_training`), writing into that fold's `output_dir`:

- `test_metrics.json` — accuracy, macro/weighted precision-recall-F1, AUC-ROC, mAP, calibration error, per-class metrics
- `test_predictions.csv` — per-image prediction + per-class probabilities
- `test_classwise_report.csv`, `confusion_matrix.csv` / `.png`, `predicted_class_distribution.json`

### Re-running evaluation without retraining (`test.py`)

Use this to re-score an existing checkpoint — e.g. against a different test split, or after changing something in `evaluation/test_utils.py`.

```bash
python test.py \
  --config swint.chu.swin_tiny_block_cbam_34_chu_full_w_5fold \
  --fold 0
```

This reads `data_root/fold_0/test/` and the checkpoint at `output_dir/fold_0/best_model` (both derived from the config), rebuilds the model architecture from the config, and loads the fine-tuned weights into it — required for the CBAM models, which are saved as a raw state dict rather than a standard HuggingFace checkpoint. Override either with `--data_root`, `--output_dir`, `--checkpoint`, or `--test_split`.

### Aggregating across folds

```bash
# Collapse 3-class predictions into melanoma-vs-other binary metrics, per fold + mean±std
python eval_melanoma_binary.py /path/to/output_dir

# Apply the focal posterior (Ψ) transform to saved test_predictions.csv and recompute metrics
python psi_transform.py --input_root /path/to/output_dir --folds 5 --num_classes 3
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lee2026uveal,
  title     = {Comparing Deep Architectures for Uveal Melanoma Classification
               in Ultra-Widefield Fundus Images},
  author    = {Lee, Haehyun and Nahon-Esteve, Sacha and Lingrand, Diane and
               Precioso, Frederic and Mattei, Pierre-Alexandre and
               Maschi, Celia and Baillif, St\'{e}phanie},
  booktitle = {MICCAI Workshop on Ophthalmic Medical Image Analysis (OMIA)},
  year      = {2026}
}
```

---

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
