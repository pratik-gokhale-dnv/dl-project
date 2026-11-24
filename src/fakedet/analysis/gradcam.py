"""Grad-CAM (own implementation) of the "fake" logit for the detectors, per generator.

For model `m` trained on the reference run, we compute Grad-CAM maps for fake (and real) test crops of
every generator and report: the mean map per generator, example overlays, and how similar the
mean maps are across generators (cosine similarity) - i.e. does the detector look at the same
evidence for every generator? ViT models use the patch tokens of the last block (reshaped to a grid).

Outputs (results/gradcam/): <model>_maps.npz, <model>_examples.png, similarity.csv, stats.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fakedet.analysis.common import analysis_dir, head_per_label, load_crop, reference_run, trained
from fakedet.config import Config, generators, title
from fakedet.data.build import load_manifest, subset
from fakedet.models.base import Detector
from fakedet.train import load_trained
from fakedet.utils import get_device, log


def grad_cam(model: Detector, x: torch.Tensor) -> torch.Tensor:
    """x: float (B,3,H,W) in [0,1]  ->  maps (B,H,W) in [0,1], explaining the (fake) logit."""
    layer, reshape = model.cam_target()
    store = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # keeps the graph alive through frozen backbones
            logit = model(x)
            logit.sum().backward()
    finally:
        h.remove()
    act, grad = store["act"].detach(), store["grad"]
    if reshape:
        act, grad = reshape(act), reshape(grad)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo, hi = cam.amin(dim=(1, 2), keepdim=True), cam.amax(dim=(1, 2), keepdim=True)
    return ((cam - lo) / (hi - lo + 1e-8)).detach()


def _cams(cfg: Config, model: Detector, paths: list[str], device, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    imgs, cams = [], []
    for i in range(0, len(paths), batch):
        crops = [np.asarray(load_crop(cfg, p)) for p in paths[i:i + batch]]
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(device)
        cams.append(grad_cam(model, x).cpu().numpy())
        imgs.append(np.stack(crops))
    return np.concatenate(imgs), np.concatenate(cams)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cam_stats(m: np.ndarray) -> dict:
    """Where (centre of mass, as fraction of the image) and how concentrated (share of mass in the top 10% pixels)."""
    h, w = m.shape
    p = m / (m.sum() + 1e-12)
    ys, xs = np.indices(m.shape)
    top = np.sort(p.ravel())[::-1][: max(1, p.size // 10)].sum()
    return {"center_x": float((p * xs).sum() / w), "center_y": float((p * ys).sum() / h),
            "top10_mass": float(top), "entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(p.size))}


def run_gradcam(cfg: Config) -> None:
    a = cfg.analysis.gradcam
    out = analysis_dir(cfg, "gradcam")
    device = get_device(cfg.project.device)
    manifest = load_manifest(cfg, a.variant)
    sim_rows, stat_rows = [], []
    for name in a.models:
        run = reference_run(cfg, name)
        if not trained(cfg, run):
            log.warning("gradcam: %s not trained, skipping", run)
            continue
        model = load_trained(cfg, run, device)
        maps, examples = {}, {}
        for g in generators(cfg):
            df = head_per_label(subset(manifest, g, "test"), a.n_images)
            for label, key in ((1, "fake"), (0, "real")):
                paths = df[df.label == label].path.tolist()
                if not paths:
                    continue
                imgs, cams = _cams(cfg, model, paths, device)
                maps[f"{g}_{key}"] = cams.mean(0)
                if label == 1:
                    examples[g] = (imgs[: a.n_examples], cams[: a.n_examples])
        np.savez_compressed(out / f"{name}_maps.npz", **maps)
        gens = [g for g in generators(cfg) if f"{g}_fake" in maps]
        for g1 in gens:
            for g2 in gens:
                sim_rows.append({"model": name, "gen_a": g1, "gen_b": g2,
                                 "cosine": cosine(maps[f"{g1}_fake"], maps[f"{g2}_fake"])})
            stat_rows.append({"model": name, "generator": g1, **cam_stats(maps[f"{g1}_fake"])})
        _plot_examples(cfg, name, examples, maps, out)
        log.info("gradcam: %s done", name)
    pd.DataFrame(sim_rows).to_csv(out / "similarity.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out / "stats.csv", index=False)


def _plot_examples(cfg: Config, name: str, examples: dict, maps: dict, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gens = list(examples)
    n = max(len(v[0]) for v in examples.values())
    fig, ax = plt.subplots(len(gens), n + 1, figsize=(1.9 * (n + 1), 1.9 * len(gens)), squeeze=False)
    for i, g in enumerate(gens):
        imgs, cams = examples[g]
        for j in range(n + 1):
            ax[i, j].axis("off")
        for j in range(len(imgs)):
            ax[i, j].imshow(imgs[j])
            ax[i, j].imshow(cams[j], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax[i, n].imshow(maps[f"{g}_fake"], cmap="jet")
        ax[i, 0].set_title(title(cfg, g), fontsize=6, loc="left")
    ax[0, n].set_title("mean CAM", fontsize=6)
    fig.suptitle(f"Grad-CAM of fake images - {cfg.models[name].get('title', name)}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_examples.png", dpi=110)
    plt.close(fig)
