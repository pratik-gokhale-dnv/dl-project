"""UnivFD-style detector (Ojha et al., CVPR 2023): frozen CLIP ViT-L/14 + a small trainable head.

`layer` selects the features fed to the head:
    "final" -> CLIP's projected image embedding (what UnivFD uses)
    int k   -> CLS token after transformer block k (intermediate layers)
`head` is "linear" or "mlp" (2 layers). The backbone never trains, so its features can be cached.
"""
from __future__ import annotations

import torch
from torch import nn

from fakedet.models.base import Detector, Normalize, tokens_to_grid

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_head(dim: int, kind: str, hidden: int = 512, dropout: float = 0.1) -> nn.Module:
    norm = nn.LayerNorm(dim, elementwise_affine=False)  # layer features have very different scales
    if kind == "linear":
        return nn.Sequential(norm, nn.Linear(dim, 1))
    if kind == "mlp":
        return nn.Sequential(norm, nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                             nn.Linear(hidden, 1))
    raise ValueError(f"head must be 'linear' or 'mlp', got {kind!r}")


class CLIPDetector(Detector):
    frozen_backbone = True

    def __init__(self, hf_name: str = "openai/clip-vit-large-patch14", layer="final", head: str = "linear",
                 pretrained: bool = True, random_config: dict | None = None, img_size: int = 224,
                 mlp_hidden: int = 512, random_seed: int = 0):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        if pretrained:
            self.backbone = CLIPVisionModelWithProjection.from_pretrained(hf_name)
            self.backbone_id = hf_name
        else:  # tiny random backbone (smoke tests); fixed seed so cached features stay valid
            conf = CLIPVisionConfig(image_size=img_size, **(random_config or {}))
            with torch.random.fork_rng():
                torch.manual_seed(random_seed)
                self.backbone = CLIPVisionModelWithProjection(conf)
            self.backbone_id = f"random:{sorted((random_config or {}).items())}:{img_size}:{random_seed}"
        self.backbone.requires_grad_(False).eval()
        conf = self.backbone.config
        self.layer = layer if layer == "final" else int(layer)
        if self.layer != "final" and not 1 <= self.layer <= conf.num_hidden_layers:
            raise ValueError(f"layer must be 'final' or in [1, {conf.num_hidden_layers}]")
        self.interpolate = img_size != conf.image_size
        self.norm = Normalize(CLIP_MEAN, CLIP_STD)
        dim = conf.projection_dim if self.layer == "final" else conf.hidden_size
        self.head = make_head(dim, head, mlp_hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen: no dropout / no state changes
        return self

    def extract(self, x: torch.Tensor, layers: list) -> dict:
        """Features of several layers in one backbone pass: {layer: (B, D)}."""
        out = self.backbone(pixel_values=self.norm(x), output_hidden_states=True,
                            interpolate_pos_encoding=self.interpolate)
        return {l: out.image_embeds if l == "final" else out.hidden_states[l][:, 0] for l in layers}

    def embed(self, x):
        return self.extract(x, [self.layer])[self.layer]

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)

    def cam_target(self):
        k = self.backbone.config.num_hidden_layers if self.layer == "final" else self.layer
        return self.backbone.vision_model.encoder.layers[k - 1].layer_norm1, tokens_to_grid(1)

    def trainable_state(self) -> dict:
        return self.head.state_dict()

    def load_trainable_state(self, state: dict) -> None:
        self.head.load_state_dict(state)
