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
