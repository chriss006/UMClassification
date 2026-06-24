CONFIG = {
    "seed": 42,
    "data_root": "/home/halee/datasets/UWF",
    "output_dir": "/home/halee/outputs/EfficientNetv2/single_cbam_uwf_full",

    "model_name": "torchvision/efficientnet_v2_s",
    "pretrained_checkpoint": "IMAGENET1K_V1",

    "use_cbam": True,
    "cbam_mode": "final",
    "cbam_reduction_ratio": 16,
    "cbam_kernel_size": 7,

    "freeze_backbone": False,
    "freeze_except_keywords": None,

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],

    "augmentation": True,
    "aug_rotation": 30,
    "aug_brightness": 0.3,
    "aug_contrast": 0.3,
    "aug_saturation": 0.1,
    "aug_translate": 0.1,
    "aug_shear": 5,

    "loss_type": "ce",
    "learning_rate": 1e-5,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 50,
    "weight_decay": 0.05,
    "num_workers": 4,

    "logging_steps": 10,
    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
    "save_total_limit": 2,

    "early_stopping_patience": 8,
    "early_stopping_threshold": 0.0,
    "resume_from_checkpoint": None,
}