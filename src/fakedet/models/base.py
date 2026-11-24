"""Detector interface shared by every model.

Every detector takes a float image batch in [0, 1] of shape (B, 3, H, W) and returns one logit per
image (> 0 means "fake"). Normalisation lives inside the model, so datasets, Grad-CAM, JPEG
perturbations and NPR's residual all see/produce the same input space.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Detector(nn.Module):
    #: True when the backbone is frozen and only `self.head` is trained (features can be cached)
    frozen_backbone = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) -> (B,)
        raise NotImplementedError

    def embed(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) -> (B,D) pre-logit features
        raise NotImplementedError

    def cam_target(self) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor] | None]:
        """(layer whose activations Grad-CAM uses, reshape fn mapping its output to (B,C,h,w))."""
        raise NotImplementedError

    def trainable_state(self) -> dict:
        return self.state_dict()

    def load_trainable_state(self, state: dict) -> None:
        self.load_state_dict(state)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def tokens_to_grid(prefix_tokens: int = 1) -> Callable[[torch.Tensor], torch.Tensor]:
    """ViT token sequence (B, P+N, C) -> (B, C, h, w), dropping the P prefix (class) tokens."""
    def fn(t: torch.Tensor) -> torch.Tensor:
        t = t[:, prefix_tokens:]
        n = t.shape[1]
        side = int(round(n ** 0.5))
        return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)
    return fn
