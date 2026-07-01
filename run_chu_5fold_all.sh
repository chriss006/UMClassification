#!/bin/bash
# Same 10-experiment CHU Block-Level CBAM suite as run_chu_re_all.sh, but on the
# 5-fold split (CHU_224_5fold, built from the corrected CHU_224_re patient pool).
# UWF backbone checkpoints only go up to fold2, so CHU fold_3/fold_4 reuse UWF
# fold0/fold1 via symlinks at /home/halee/outputs/SwinT/uwf/*/fold{3,4}.
# Runs sequentially on one GPU. Safe to re-run individual lines by hand if one
# step fails.
set -e

cd /home/halee/projects/UMClassification

echo "=== [1/10] Baseline: EfficientNetV2 + Single CBAM (CE) ==="
python3 finetune.py --config efficientnet.efficientnetv2_s_single_cbam_chu_5fold --use_predefined_folds

echo "=== [2/10] Baseline: SwinT Full fine-tuning (CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_chu_full_5fold --use_predefined_folds

echo "=== [3/10] Block CBAM: Stage 4 (frozen, CBAM+Classifier only) (CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_chu_5fold --use_predefined_folds

echo "=== [4/10] Block CBAM: Stage 4, Full fine-tuning (CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_chu_full_5fold --use_predefined_folds

echo "=== [5/10] Block CBAM: Stage 4, Full fine-tuning (Weighted CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_chu_full_w_5fold --use_predefined_folds

echo "=== [6/10] Block CBAM: Stage 3&4, Full fine-tuning (CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_34_chu_full_5fold --use_predefined_folds

echo "=== [7/10] Block CBAM: Stage 3&4, Full fine-tuning (Weighted CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_34_chu_full_w_5fold --use_predefined_folds

echo "=== [8/10] Block CBAM: All Stages, Full fine-tuning (CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_all_chu_full_5fold --use_predefined_folds

echo "=== [9/10] Block CBAM: All Stages, Full fine-tuning (Weighted CE) ==="
python3 finetune.py --config swint.chu.swin_tiny_block_cbam_all_chu_full_w_5fold --use_predefined_folds

echo "=== [10/10] Baseline: RETFound CFP, Full fine-tuning (CE) ==="
cd /home/halee/projects/RETFound
for fold in 0 1 2 3 4; do
  python3 main_finetune.py \
    --batch_size 16 --accum_iter 2 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --smoothing 0 --loss ce \
    --nb_classes 3 \
    --num_workers 8 \
    --early_stop_metric f1_macro --early_stop_patience 10 \
    --data_path /home/halee/datasets/CHU_224_5fold/fold_${fold} \
    --output_dir /home/halee/outputs/RETFound/chu_5fold/cfp_full \
    --task fold_${fold} \
    --finetune /home/halee/weights/RETFound_cfp_weights.pth
done

echo "=== All 10 experiments complete (5-fold) ==="
