import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput

RETFOUND_REPO_PATH = "/home/halee/projects/RETFound"
if RETFOUND_REPO_PATH not in sys.path:
    sys.path.insert(0, RETFOUND_REPO_PATH)

import models_vit
from util.pos_embed import interpolate_pos_embed

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_


class RETFoundClassifier(nn.Module):
    """RETFound ViT-L/16 backbone -> classifier head, full fine-tuning."""

    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, pixel_values, labels=None, **kwargs):
        features = self.vit.forward_features(pixel_values)
        logits = self.vit.head(features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


def build_retfound_model(checkpoint_path, num_classes, global_pool=False, drop_path_rate=0.1):
    model = models_vit.vit_large_patch16(
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"\nLoaded RETFound checkpoint from {checkpoint_path}")
    print(f"missing_keys: {msg.missing_keys}")

    trunc_normal_(model.head.weight, std=2e-5)

    return RETFoundClassifier(model)
