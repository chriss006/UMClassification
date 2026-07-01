import torch

CONFIG = {
    "data_root": "/home/halee/datasets/CHU_224_re",
    "output_dir": "/home/halee/outputs/SwinT/chu_re/block_cbam(4)_full",
    "model_name": "microsoft/swin-tiny-patch4-window7-224",
    "pretrained_checkpoint": "/home/halee/outputs/SwinT/uwf/swint_ce/fold{fold}/best_model",
    "cbam_init_checkpoint": "/home/halee/outputs/SwinT/uwf/block_cbam(4)/fold{fold}/best_model",

    "num_labels": 3,

    # CBAM settings (must match the pretrained checkpoint's architecture)
    "use_cbam": True,
    "cbam_mode": "block",
    "cbam_stage_indices": [3],
    "cbam_reduction_ratio": 16,
    "cbam_kernel_size": 7,

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],

    "seed": 42,
    "num_train_epochs": 50,
    "learning_rate": 2e-5,
    "weight_decay": 1e-4,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_workers": 2,
    "save_total_limit": 2,
    "logging_steps": 10,

    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.001,

    # Augmentation (train only) — important given CHU's small size
    "augmentation": True,
    "aug_rotation": 30,
    "aug_brightness": 0.3,
    "aug_contrast": 0.3,
    "aug_saturation": 0.1,
    "aug_translate": 0.1,
    "aug_shear": 5,

    "loss_type": "ce",
    "apply_posterior_transformation": False,

    "resume_from_checkpoint": None,

    # Full finetuning: nothing frozen, all params trainable.
    "freeze_backbone": False,
    "freeze_except_keywords": None,
}
