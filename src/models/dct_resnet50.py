"""DCT ResNet-50 (Person B): same backbone as rgb_resnet50, frequency-domain input.

Input: [0, 1] RGB, B x 3 x H x W. Each channel goes through an orthonormal 2D
DCT-II, then log-magnitude, following Frank et al. (2020). Log scaling matters:
raw coefficients span several orders of magnitude and the high-frequency ones,
where generator artifacts live, would otherwise be swamped by the DC term.
"""

import math

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models import register


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix D, so that D @ x @ D.T is the 2D DCT of x."""
    k = torch.arange(n).unsqueeze(1).double()
    i = torch.arange(n).unsqueeze(0).double()
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
    d[0] /= math.sqrt(2)
    return d.float()


class DCTResNet50(nn.Module):
    def __init__(self, num_outputs: int, size: int = 256, pretrained: bool = True):
        super().__init__()
        self.register_buffer("D", dct_matrix(size), persistent=False)
        # Normalizes log-DCT values per channel with running train-set statistics.
        # Frank et al. standardize per coefficient instead; worth comparing later.
        self.input_norm = nn.BatchNorm2d(3)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_outputs)

    def forward(self, x):
        z = self.D @ (x * 255) @ self.D.T
        z = torch.log(z.abs() + 1e-3)
        return self.backbone(self.input_norm(z))


@register("dct_resnet50")
def build(num_outputs: int, size: int = 256, pretrained: bool = True):
    return DCTResNet50(num_outputs, size, pretrained)
