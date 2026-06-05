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
    cbam_mode = config.get("cbam_mode", "final")
    if cbam_mode == "stage":
        return _build_stage_cbam_model(config, id2label, label2id)
    if cbam_mode == "block":
        return _build_block_cbam_model(config, id2label, label2id)

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
        hidden_size = full_model.swin.num_features
        for p in backbone.parameters():
            p.requires_grad_(False)
        return SwinWithCBAM(backbone, num_labels, hidden_size, reduction_ratio, kernel_size)

    else:
        raise ValueError(f"CBAM not implemented for model: {config['model_name']}")


def _build_stage_cbam_model(config, id2label, label2id):
    from core.cbam import ConvNextWithStageCBAM, SwinWithStageCBAM

    model_name = config["model_name"].lower()
    num_labels = len(id2label)
    pretrained_checkpoint = config["pretrained_checkpoint"]
    stage_indices = config.get("cbam_stage_indices", [2, 3])
    reduction_ratio = config.get("cbam_reduction_ratio", 16)
    kernel_size = config.get("cbam_kernel_size", 7)

    if "convnext" in model_name:
        full_model = ConvNextForImageClassification.from_pretrained(
            pretrained_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        # hidden_sizes: [96, 192, 384, 768] for convnext-tiny
        hidden_sizes = full_model.config.hidden_sizes
        stage_cbam_channels = {i: hidden_sizes[i] for i in stage_indices}
        return ConvNextWithStageCBAM(full_model, num_labels, stage_cbam_channels,
                                     reduction_ratio, kernel_size)

    elif "swin" in model_name:
        full_model = SwinForImageClassification.from_pretrained(
            pretrained_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        embed_dim = full_model.config.embed_dim          # 96 for swin-tiny
        num_stages = len(full_model.config.depths)       # 4 for swin-tiny
        # Stage i output channels after patch merging:
        #   stages with downsample (i < num_stages-1): embed_dim * 2^(i+1)
        #   last stage (no downsample):                embed_dim * 2^i
        stage_cbam_channels = {
            i: int(embed_dim * 2 ** (i + 1 if i < num_stages - 1 else i))
            for i in stage_indices
        }
        return SwinWithStageCBAM(full_model, num_labels, stage_cbam_channels,
                                  reduction_ratio, kernel_size)

    else:
        raise ValueError(f"Stage CBAM not implemented for model: {config['model_name']}")


def _build_block_cbam_model(config, id2label, label2id):
    from core.cbam import SwinWithBlockCBAM

    model_name = config["model_name"].lower()
    num_labels = len(id2label)
    pretrained_checkpoint = config["pretrained_checkpoint"]
    stage_indices = config.get("cbam_stage_indices", [1, 2])
    reduction_ratio = config.get("cbam_reduction_ratio", 16)
    kernel_size = config.get("cbam_kernel_size", 7)

    if "swin" in model_name:
        full_model = SwinForImageClassification.from_pretrained(
            pretrained_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        return SwinWithBlockCBAM(full_model, num_labels, stage_indices,
                                  reduction_ratio, kernel_size)

    else:
        raise ValueError(f"Block CBAM is only implemented for SwinT: {config['model_name']}")