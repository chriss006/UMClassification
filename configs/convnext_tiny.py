CONFIG = {
    "data_root": "/home/halee/datasets/UWF",
    "output_dir": "/home/halee/outputs/convnext_tiny",
    "model_name": "facebook/convnext-tiny-224",
    "num_labels": 6,

    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],

    "seed": 42,
    "num_train_epochs": 50,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_workers": 2,
    "fp16": torch.cuda.is_available(),
    "save_total_limit": 2,
    "logging_steps": 10,

    "metric_for_best_model": "eval_mAP",
    "greater_is_better": True,
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.001,

    "focal_gamma": 2.0,
    "focal_alpha": None,

    "resume_from_checkpoint": None,  # 예: "/home/halee/outputs/convnext_tiny/checkpoint-306"
}
