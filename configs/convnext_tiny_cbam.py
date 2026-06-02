import torch

CONFIG = {
    "data_root": "/Volumes/Extreme SSD/MAASAI/dataset/UWF",
    "output_dir": "/Volumes/Extreme SSD/MAASAI/finetune_UWF/ConvNext_CBAM",

    # Base model name (used only to load the image processor)
    "model_name": "facebook/convnext-tiny-224",
    # Path to the already-trained ConvNext checkpoint (saved by trainer.save_model)
    "pretrained_checkpoint": "/Volumes/Extreme SSD/MAASAI/finetune_UWF/ConvNext/best_model",

    "num_labels": 6,

    # CBAM settings
    "use_cbam": True,
    "cbam_reduction_ratio": 16,   # channel attention bottleneck ratio
    "cbam_kernel_size": 7,        # spatial attention conv kernel

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],

    "seed": 42,
    "num_train_epochs": 30,
    # Higher LR is fine: only CBAM + LayerNorm + classifier are trainable
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

    "loss_type": "focal",
    "focal_gamma": 2.0,
    "focal_alpha": None,
    "apply_posterior_transformation": False,

    "resume_from_checkpoint": None,
    # Backbone is frozen inside build_cbam_model; do NOT set freeze_backbone=True
    # (that helper only knows about "classifier"/"head"/"fc" keywords and would
    #  accidentally freeze the CBAM module as well)
    "freeze_backbone": False,
}
