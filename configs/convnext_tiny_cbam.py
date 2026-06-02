import torch

CONFIG = {
    "data_root": "/home/halee/datasets/UWF",
    "output_dir": "/home/halee/outputs/ConvNext/cbam",
    "model_name": "facebook/convnext-tiny-224",
    "pretrained_checkpoint": "/home/halee/outputs/ConvNext/convnext_ce_3fold/fold{fold}/best_model",

    "num_labels": 6,

    # CBAM settings
    "use_cbam": True,
    "cbam_reduction_ratio": 16,
    "cbam_kernel_size": 7,

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],

    "seed": 42,
    "num_train_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_workers": 2,
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
    "focal_gamma": 2.0,
    "focal_alpha": None,
    "apply_posterior_transformation": False,

    "resume_from_checkpoint": None,
    "freeze_backbone": False,
}
