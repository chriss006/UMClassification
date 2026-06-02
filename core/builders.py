# model_builder.py

from transformers import (
    ConvNextForImageClassification,
    SwinForImageClassification,
    AutoModelForImageClassification,
)


def build_model(config, id2label, label2id):
    if config.get("use_cbam", False):
        return _build_cbam_model(config, id2label, label2id)

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


def _build_cbam_model(config, id2label, label2id):
    from core.cbam import ConvNextWithCBAM, SwinWithCBAM

    model_name = config["model_name"].lower()
    num_labels = len(id2label)
    pretrained_checkpoint = config["pretrained_checkpoint"]
    reduction_ratio = config.get("cbam_reduction_ratio", 16)
    kernel_size = config.get("cbam_kernel_size", 7)

    if "convnext" in model_name:
        full_model = ConvNextForImageClassification.from_pretrained(pretrained_checkpoint)
        backbone = full_model.convnext
        hidden_size = full_model.config.hidden_sizes[-1]
        for p in backbone.parameters():
            p.requires_grad_(False)
        return ConvNextWithCBAM(backbone, num_labels, hidden_size, reduction_ratio, kernel_size)

    elif "swin" in model_name:
        full_model = SwinForImageClassification.from_pretrained(pretrained_checkpoint)
        backbone = full_model.swin
        # num_features = hidden_size * 2^(num_stages-1), e.g. 96*8=768 for swin-tiny
        hidden_size = full_model.num_features
        for p in backbone.parameters():
            p.requires_grad_(False)
        return SwinWithCBAM(backbone, num_labels, hidden_size, reduction_ratio, kernel_size)

    else:
        raise ValueError(f"CBAM not implemented for model: {config['model_name']}")