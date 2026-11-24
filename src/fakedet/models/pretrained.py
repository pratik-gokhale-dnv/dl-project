"""ImageNet pre-trained detectors (ResNet-50, DeiT-S) fine-tuned end to end, via timm."""
from __future__ import annotations

import timm

from fakedet.models.base import Detector, Normalize, tokens_to_grid


class TimmDetector(Detector):
    def __init__(self, arch: str, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        kwargs = {"img_size": img_size} if self._is_vit(arch) and img_size != 224 else {}
        self.net = timm.create_model(arch, pretrained=pretrained, num_classes=1, **kwargs)
        cfg = self.net.pretrained_cfg
        self.norm = Normalize(cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)))
        self.is_vit = hasattr(self.net, "blocks")

    @staticmethod
    def _is_vit(arch: str) -> bool:
        return any(k in arch for k in ("vit", "deit"))

    def embed(self, x):
        feats = self.net.forward_features(self.norm(x))
        return self.net.forward_head(feats, pre_logits=True)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(1)

    def cam_target(self):
        if self.is_vit:
            return self.net.blocks[-1].norm1, tokens_to_grid(getattr(self.net, "num_prefix_tokens", 1))
        last_stage = [m for n, m in self.net.named_children() if n.startswith("layer")][-1]
        return last_stage[-1], None
