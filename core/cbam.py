import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = x.mean(dim=[-2, -1])                    # [B, C]
        max_ = x.flatten(2).max(dim=2).values         # [B, C]
        attention = self.sigmoid(self.shared_mlp(avg) + self.shared_mlp(max_))
        return x * attention[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.max(dim=1, keepdim=True).values
        attention = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvNextWithCBAM(nn.Module):
    """Frozen ConvNext backbone → CBAM → LayerNorm → Linear classifier."""

    def __init__(self, backbone, num_classes: int, hidden_size: int,
                 reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(hidden_size, reduction_ratio, kernel_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor, labels=None, **kwargs):
        with torch.no_grad():
            out = self.backbone(pixel_values)
        # last_hidden_state: [B, C, H, W] (ConvNext keeps spatial dims)
        features = out.last_hidden_state.detach()

        features = self.cbam(features)
        pooled = features.mean(dim=[-2, -1])   # [B, C]
        pooled = self.layernorm(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


class ConvNextWithStageCBAM(nn.Module):
    """CBAM inserted after specified ConvNext encoder stages via forward hooks (full fine-tuning).

    stage_cbam_channels: {stage_idx: num_channels}
    - CBAM after Stage 3 (384ch) and Stage 4 (768ch).
    """

    def __init__(self, backbone_model, num_classes: int, stage_cbam_channels: dict,
                 reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone_model  # ConvNextForImageClassification

        self.cbam_modules = nn.ModuleDict({
            str(idx): CBAM(ch, reduction_ratio, kernel_size)
            for idx, ch in stage_cbam_channels.items()
        })

        # Register a forward hook on each target stage
        self._hook_handles = []
        for idx in stage_cbam_channels:
            stage = self.backbone.convnext.encoder.stages[idx]
            handle = stage.register_forward_hook(self._make_hook(str(idx)))
            self._hook_handles.append(handle)

    def _make_hook(self, stage_key: str):
        def hook(module, input, output):
            # output may be a tensor or a tuple depending on transformers version
            if isinstance(output, (tuple, list)):
                x = self.cbam_modules[stage_key](output[0])
                return (x,) + tuple(output[1:])
            else:
                return self.cbam_modules[stage_key](output)
        return hook

    def forward(self, pixel_values: torch.Tensor, labels=None, **kwargs):
        outputs = self.backbone(pixel_values)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


class SwinWithStageCBAM(nn.Module):
    """CBAM inserted after specified Swin encoder stages via forward hooks (full fine-tuning).

    Swin stage output is [B, seq_len, C]. The hook reshapes to [B, C, H, W],

    stage_cbam_channels: {stage_idx: num_channels_after_patch_merging}
      e.g. {1: 384, 2: 768} for swin-tiny (Stage 3 H/16 and Stage 4 H/32).
    """

    def __init__(self, backbone_model, num_classes: int, stage_cbam_channels: dict,
                 reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone_model  # SwinForImageClassification

        self.cbam_modules = nn.ModuleDict({
            str(idx): CBAM(ch, reduction_ratio, kernel_size)
            for idx, ch in stage_cbam_channels.items()
        })

        self._hook_handles = []
        for idx in stage_cbam_channels:
            # Swin stages live at backbone.swin.encoder.layers
            stage = self.backbone.swin.encoder.layers[idx]
            handle = stage.register_forward_hook(self._make_hook(str(idx)))
            self._hook_handles.append(handle)

    def _make_hook(self, stage_key: str):
        def hook(module, input, output):
            # Swin stage returns (downsampled, before_downsample, output_dims, ...)
            hidden_states = output[0]              # [B, seq_len, C]
            B, seq_len, C = hidden_states.shape
            H = W = int(math.isqrt(seq_len))
            x = hidden_states.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
            x = self.cbam_modules[stage_key](x)
            x = x.permute(0, 2, 3, 1).reshape(B, seq_len, C)         # [B, seq_len, C]
            return (x,) + output[1:]
        return hook

    def forward(self, pixel_values: torch.Tensor, labels=None, **kwargs):
        outputs = self.backbone(pixel_values)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


class SwinWithBlockCBAM(nn.Module):
    """Block-Level CBAM for SwinT (Stage 3, 4 only).

    Inserts CAM before W-MSA (even-indexed blocks) and
    SAM before SW-MSA (odd-indexed blocks) via forward pre-hooks
    on each SwinLayer's attention module.

    Block channels (within blocks, before patch merging):
      layers[1]: embed_dim * 2^1 = 192  (Stage 3, 28x28)
      layers[2]: embed_dim * 2^2 = 384  (Stage 4, 14x14)
    """

    def __init__(self, backbone_model, num_classes: int, target_stage_indices: list,
                 reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone_model

        embed_dim = backbone_model.config.embed_dim  # 96 for swin-tiny

        self.cam_modules = nn.ModuleDict()
        self.sam_modules = nn.ModuleDict()
        self._hook_handles = []

        for stage_idx in target_stage_indices:
            block_channels = int(embed_dim * 2 ** stage_idx)
            stage = backbone_model.swin.encoder.layers[stage_idx]

            for block_idx, block in enumerate(stage.blocks):
                key = f"s{stage_idx}_b{block_idx}"
                if block_idx % 2 == 0:  # W-MSA block → CAM
                    self.cam_modules[key] = ChannelAttention(block_channels, reduction_ratio)
                    handle = block.attention.register_forward_pre_hook(
                        self._make_cam_hook(key)
                    )
                else:  # SW-MSA block → SAM
                    self.sam_modules[key] = SpatialAttention(kernel_size)
                    handle = block.attention.register_forward_pre_hook(
                        self._make_sam_hook(key)
                    )
                self._hook_handles.append(handle)

    def _make_cam_hook(self, key: str):
        def hook(module, args):
            hidden_states = args[0]   # [B, seq_len, C] — after layernorm_before
            B, seq_len, C = hidden_states.shape
            H = W = int(math.isqrt(seq_len))
            x = hidden_states.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            x = self.cam_modules[key](x)
            x = x.permute(0, 2, 3, 1).reshape(B, seq_len, C)        # [B, seq_len, C]
            return (x,) + args[1:]
        return hook

    def _make_sam_hook(self, key: str):
        def hook(module, args):
            hidden_states = args[0]   # [B, seq_len, C] — after layernorm_before
            B, seq_len, C = hidden_states.shape
            H = W = int(math.isqrt(seq_len))
            x = hidden_states.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            x = self.sam_modules[key](x)
            x = x.permute(0, 2, 3, 1).reshape(B, seq_len, C)        # [B, seq_len, C]
            return (x,) + args[1:]
        return hook

    def forward(self, pixel_values: torch.Tensor, labels=None, **kwargs):
        outputs = self.backbone(pixel_values)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)


class SwinWithCBAM(nn.Module):
    """Frozen Swin backbone → reshape to 2D → CBAM → LayerNorm → Linear classifier."""

    def __init__(self, backbone, num_classes: int, hidden_size: int,
                 reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(hidden_size, reduction_ratio, kernel_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor, labels=None, **kwargs):
        with torch.no_grad():
            out = self.backbone(pixel_values)
        # last_hidden_state: [B, seq_len, C]  (Swin flattens spatial dims)
        features = out.last_hidden_state.detach()
        B, seq_len, C = features.shape
        H = W = int(math.isqrt(seq_len))
        # Reshape back to 2D spatial: [B, C, H, W]
        features = features.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        features = self.cbam(features)
        pooled = features.mean(dim=[-2, -1])   # [B, C]
        pooled = self.layernorm(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)
