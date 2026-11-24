import pytest
import torch

from fakedet.models import build_model, clip_layers_needed
from fakedet.models.scratch import NPR


@pytest.mark.parametrize("name", ["cnn", "npr", "npr_rgb", "resnet50", "deit_s", "clip_final_linear",
                                  "clip_mid_mlp"])
def test_forward_embed_gradcam_target(cfg, name):
    model = build_model(cfg, name).eval()
    x = torch.rand(2, 3, 64, 64)
    logits, emb = model(x), model.embed(x)
    assert logits.shape == (2,) and emb.ndim == 2 and emb.shape[0] == 2
    layer, reshape = model.cam_target()
    acts = {}
    h = layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    model(x.clone().requires_grad_(True))
    h.remove()
    a = reshape(acts["a"]) if reshape else acts["a"]
    assert a.ndim == 4 and a.shape[0] == 2


def test_npr_residual_is_zero_on_blocky_images():
    flat = torch.rand(1, 3, 8, 8).repeat_interleave(2, 2).repeat_interleave(2, 3)  # constant 2x2 blocks
    assert NPR.npr(flat).abs().max() == 0
    assert NPR.npr(torch.rand(1, 3, 16, 16)).abs().max() > 0


def test_clip_frozen_and_head_only_state(cfg):
    m = build_model(cfg, "clip_mid_mlp")
    assert m.frozen_backbone and m.num_trainable() == sum(p.numel() for p in m.head.parameters())
    m.train()
    assert not m.backbone.training
    assert set(m.trainable_state()) == set(m.head.state_dict())
    assert clip_layers_needed(cfg) == [16, "final"] or clip_layers_needed(cfg) == ["final", 16]


def test_clip_layers_and_extract(cfg):
    m = build_model(cfg, "clip_final_linear").eval()
    f = m.extract(torch.rand(2, 3, 64, 64), ["final", 16, 24])
    assert f["final"].shape == (2, 16) and f[16].shape == (2, 32) and f[24].shape == (2, 32)
