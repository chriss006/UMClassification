import torch

CONFIG = {
    "data_root": "/home/halee/datasets/UWF_224",
    "output_dir": "/home/halee/outputs/ConvNext/stage_cbam",
    "model_name": "facebook/convnext-tiny-224",
    "pretrained_checkpoint": "/home/halee/outputs/ConvNext/convnext_ce_3fold/fold{fold}/best_model",

    "num_labels": 6,

    # CBAM settings
    "use_cbam": True,
    "cbam_mode": "stage",           # insert CBAM inside the encoder
    "cbam_stage_indices": [2, 3],   # after Stage 3 (384ch) and Stage 4 (768ch)
    "cbam_reduction_ratio": 16,
    "cbam_kernel_size": 7,

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],

    "seed": 42,
    "num_train_epochs": 50,
    # Lower LR: full fine-tuning (backbone is also updated)
    "learning_rate": 2e-5,
    "weight_decay": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_workers": 8,
    "save_total_limit": 2,
    "logging_steps": 10,

    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.001,

    # Augmentation (train only)
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
    "freeze_backbone": False,
}
