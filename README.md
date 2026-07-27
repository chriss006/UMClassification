# Comparing Deep Architectures for Uveal Melanoma Classification in Ultra-Widefield Fundus Images

> **MICCAI 2026 Workshop OMIA** · Submission #36 · CC BY 4.0

**Authors:** Haehyun Lee, Sacha Nahon-Esteve, Diane Lingrand, Frederic Precioso, Pierre-Alexandre Mattei, Celia Maschi, Stéphanie Baillif

---

## Overview

We systematically compare three deep learning architectures for uveal melanoma (UM) classification from single-modality ultra-widefield (UWF) color fundus images:

- **RETFound** (ViT-L/16, CFP pre-trained) — full fine-tuning
- **EfficientNetV2-S + CBAM** — UWF pre-trained with a single appended CBAM module
- **Swin Transformer Tiny + Block-Level CBAM** — UWF pre-trained, CBAM injected at various stages

Two classification tasks are evaluated on a private clinical CHU dataset:
- **3-class**: Uveal Melanoma (UM) vs. IMT vs. Naevi
- **Binary**: Melanoma vs. Other (TMI + Naevi)

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

| Variant | Stage indices | # attention modules with CAM/SAM |
|---|---|---|
| Stage 4 (frozen backbone) | `[3]` | 6 (3×CAM + 3×SAM) |
| Stage 4 (full fine-tuning) | `[3]` | 6 |
| Stage 3 & 4 | `[2, 3]` | 8 (2×CAM + 2×SAM + 3×CAM + 3×SAM) |
| All Stages | `[0, 1, 2, 3]` | 12 |

Loss options: **Cross-Entropy (CE)** and **Weighted CE** (inverse-frequency weights).

---

## Repository Structure

```
UMClassification/
├── finetune.py                  # Main training script
├── test.py                      # Evaluation on test split
├── eval_melanoma_binary.py      # Binary-specific evaluation
├── psi_transform.py             # Posterior probability transform (Ψ)
├── core/
│   ├── cbam.py                  # CBAM + SwinWithBlockCBAM / SwinWithStageCBAM
│   ├── builders.py              # Model factory (SwinT, EfficientNetV2, RETFound)
│   ├── dataset.py               # ImageFolderWithPaths, collators
│   ├── losses.py                # FocalLoss, weighted CE helpers
│   ├── metrics.py               # Accuracy, macro F1, AUC-ROC, mAP
│   ├── trainers.py              # Custom HuggingFace Trainers
│   └── callbacks.py             # TrainValHistoryCallback, PrettyLogCallback
├── configs/
│   ├── swint/chu/               # SwinT Block-CBAM configs (3-class & binary)
│   ├── efficientnet/            # EfficientNetV2 + Single CBAM configs
│   └── convnext/                # ConvNext configs (ablation)
└── evaluation/
    └── test_utils.py            # run_test_and_save_outputs()
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

Install:
```bash
pip install torch torchvision transformers timm scikit-learn pandas numpy tqdm albumentations
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

Update `data_root` in the config file accordingly. Class order must be alphabetical (as used by `ImageFolder`):
- **3-class**: `melanoma=0, naevi=1, tmi=2`  → *(adjust label mapping in config)*
- **Binary**: `melanoma=0, other=1`

---

## Training

### SwinT + Block-Level CBAM (3-class, 5-fold)

```bash
# Stage 3 & 4, Weighted CE, fold 0
python finetune.py \
  --config configs/swint/chu/swin_tiny_block_cbam_34_chu_full_w_5fold.py \
  --fold 0
```

### SwinT + Block-Level CBAM (binary, 5-fold)

```bash
python finetune.py \
  --config configs/swint/chu/swin_tiny_block_cbam_34_chu_full_5fold_binary.py \
  --fold 0
```

Key config parameters:

| Parameter | Description |
|---|---|
| `cbam_stage_indices` | Swin stage indices to insert CBAM (0-indexed: 0–3) |
| `cbam_mode` | `"block"` (per-block) or `"stage"` (post-stage) |
| `loss_type` | `"ce"` or `"weighted_ce"` |
| `pretrained_checkpoint` | UWF pre-trained SwinT checkpoint path |
| `num_train_epochs` | Default: 50 |
| `early_stopping_patience` | Default: 10 |

---

## Evaluation

```bash
python test.py \
  --config configs/swint/chu/swin_tiny_block_cbam_34_chu_full_w_5fold.py \
  --fold 0 \
  --checkpoint /path/to/best_model
```

Outputs per fold: `test_metrics.json`, `test_predictions.csv` (with per-class probabilities).

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
