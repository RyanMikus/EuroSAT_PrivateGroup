"""Model builders for the challenge.

Track limits: medium <= 10M parameters, small <= 1M parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

MEDIUM_TRACK_LIMIT = 10_000_000
SMALL_TRACK_LIMIT = 1_000_000

# Pretrained RGB weights are ordered R, G, B, which are these Sentinel-2 bands.
RGB_BANDS = ["B04", "B03", "B02"]


def _adapt_first_conv(conv: nn.Conv2d, bands: list[str], pretrained: bool) -> nn.Conv2d:
    """Return a copy of ``conv`` that accepts one input channel per band.

    With pretrained weights, the RGB filters are reused for B04/B03/B02 and every other
    band starts from the mean of the RGB filters. All filters are rescaled so the layer's
    output keeps the size it had with 3 input channels.
    """
    new = nn.Conv2d(
        len(bands), conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            rgb = conv.weight  # (out, 3, k, k)
            mean = rgb.mean(dim=1)
            for i, band in enumerate(bands):
                new.weight[:, i] = rgb[:, RGB_BANDS.index(band)] if band in RGB_BANDS else mean
            new.weight.mul_(3 / len(bands))
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
    return new


def build_efficientnet_b0(bands: list[str], num_classes: int = 10, pretrained: bool = True,
                          dropout: float = 0.2) -> nn.Module:
    """EfficientNet-B0 (about 4M parameters) for multispectral input. Medium track."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    net.features[0][0] = _adapt_first_conv(net.features[0][0], bands, pretrained)
    net.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, num_classes)
    )
    return net


MODEL_BUILDERS = {
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(name: str, bands: list[str], num_classes: int = 10, **kwargs) -> nn.Module:
    """Build a model by name, see MODEL_BUILDERS."""
    return MODEL_BUILDERS[name](bands=bands, num_classes=num_classes, **kwargs)
