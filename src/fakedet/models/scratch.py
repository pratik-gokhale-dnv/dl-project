"""Detectors trained from scratch: a plain CNN and NPR (Tan et al., CVPR 2024)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from fakedet.models.base import IMAGENET_MEAN, IMAGENET_STD, Detector, Normalize


def _conv_bn_relu(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class ScratchCNN(Detector):
    """VGG-style CNN: per stage two Conv-BN-ReLU, then 2x2 max-pool (no pooling after the last stage),
    global average pooling and a linear classifier. Operates on normalised RGB."""

    def __init__(self, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        stages, c_in = [], 3
        for i, w in enumerate(widths):
            layers = [_conv_bn_relu(c_in, w), _conv_bn_relu(w, w)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*layers))
            c_in = w
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(widths[-1], 1)

    def embed(self, x):
        return self.stages(self.norm(x)).mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.embed(x)).squeeze(1)

    def cam_target(self):
        return self.stages[-1], None


class NPR(Detector):
    """Neighbouring Pixel Relationships detector.

    Input transform: NPR(x) = x - up(down(x)) with nearest-neighbour 2x down/up-sampling, i.e. the
    residual between every pixel and the top-left pixel of its 2x2 block, scaled by 2/3 (as in the
    official code). Network: ResNet-50 bottleneck blocks truncated after layer2 (stem conv 3x3/s2),
    ~1.4M parameters, random initialisation. `input="rgb"` feeds normal normalised RGB instead.
    """

    def __init__(self, input: str = "npr"):
        super().__init__()
        if input not in ("npr", "rgb"):
            raise ValueError("NPR input must be 'npr' or 'rgb'")
        self.input_mode = input
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        r = torchvision.models.resnet50(weights=None)
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.fc = nn.Linear(512, 1)

    @staticmethod
    def npr(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            raise ValueError("NPR needs even image sizes (crop_size must be even)")
        down = F.interpolate(x, scale_factor=0.5, mode="nearest")
        return x - F.interpolate(down, scale_factor=2, mode="nearest")

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.input_mode == "npr":
            with torch.autocast(device_type=x.device.type, enabled=False):  # the residual is tiny: keep fp32
                x = self.npr(x.float()) * (2.0 / 3.0)
        return x

    def embed(self, x):
        h = self.layer2(self.layer1(self.stem(self._input(x))))
        return h.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.embed(x)).squeeze(1)

    def cam_target(self):
        return self.layer2[-1], None
