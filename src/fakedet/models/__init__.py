"""Model registry: build_model(cfg, name) turns a `models:` entry of the config into a Detector."""
from __future__ import annotations

from fakedet.config import Config
from fakedet.models.base import Detector


def build_model(cfg: Config, name: str) -> Detector:
    m, crop = cfg.models[name], cfg.data.crop_size
    kind = m.type
    if kind == "cnn":
        from fakedet.models.scratch import ScratchCNN
        return ScratchCNN(widths=tuple(m.get("widths", (32, 64, 128, 256, 256))))
    if kind == "npr":
        from fakedet.models.scratch import NPR
        return NPR(input=m.get("input", "npr"))
    if kind == "timm":
        from fakedet.models.pretrained import TimmDetector
        return TimmDetector(m.arch, pretrained=m.get("pretrained", True), img_size=crop)
    if kind == "clip":
        from fakedet.models.clip import CLIPDetector
        rc = m.get("random_config")
        return CLIPDetector(hf_name=m.get("hf_name", "openai/clip-vit-large-patch14"), layer=m.get("layer", "final"),
                            head=m.get("head", "linear"), pretrained=m.get("pretrained", True),
                            random_config=rc.to_dict() if rc else None, img_size=crop,
                            mlp_hidden=m.get("mlp_hidden", 512))
    raise ValueError(f"unknown model type {kind!r}")


def clip_layers_needed(cfg: Config) -> list:
    """Union of CLIP layers requested by the configured CLIP variants (extracted in one pass)."""
    layers = {m.get("layer", "final") for m in cfg.models.values() if m.type == "clip"}
    return sorted(layers, key=lambda l: (isinstance(l, int), l if isinstance(l, int) else 0))
