# model_builder.py

from transformers import (
    ConvNextForImageClassification,
    SwinForImageClassification,
    AutoModelForImageClassification,
)


def build_model(config, id2label, label2id):
    model_name = config["model_name"].lower()
    num_labels = len(id2label)

    if "convnext" in model_name:
        model_cls = ConvNextForImageClassification
    elif "swin" in model_name:
        model_cls = SwinForImageClassification
    else:
        model_cls = AutoModelForImageClassification

    model = model_cls.from_pretrained(
        config["model_name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model