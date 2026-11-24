"""Fourier analysis of generator fingerprints (failure analysis + the spectral distance used by H4).

For every generator we average the centred log-magnitude 2-D spectrum of real and of fake test crops.
The radial (azimuthally averaged) profile of  mean_log_spectrum(fake) - mean_log_spectrum(real)  is the
generator's spectral "fingerprint"; the distance between two generators is the RMS difference of
their fingerprints. No model is involved, so this only needs the dataset.

Outputs (results/fourier/): spectra.npz, radial_profiles.csv, spectral_distance.csv, spectra.png,
radial_fingerprints.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fakedet.analysis.common import analysis_dir, head_per_label, load_crop
from fakedet.config import Config, generators, title
from fakedet.data.build import load_manifest, subset
from fakedet.utils import log


def log_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centred log-magnitude of the 2-D FFT of a (H, W) float image (mean removed)."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    return np.log(np.abs(f) + 1e-8)


def radial_profile(spec: np.ndarray) -> np.ndarray:
    """Average of a centred 2-D spectrum over rings of equal radius -> (H // 2,) profile."""
    h, w = spec.shape
    y, x = np.indices(spec.shape)
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    keep = r < h // 2
    return np.bincount(r[keep], spec[keep]) / np.maximum(np.bincount(r[keep]), 1)


def mean_spectrum(cfg: Config, paths: list[str]) -> np.ndarray:
    acc = None
    for p in paths:
        gray = np.asarray(load_crop(cfg, p).convert("L"), dtype=np.float64) / 255.0
        s = log_spectrum(gray)
        acc = s if acc is None else acc + s
    return acc / len(paths)


def fingerprint(real_spec: np.ndarray, fake_spec: np.ndarray) -> np.ndarray:
    return radial_profile(fake_spec) - radial_profile(real_spec)


def spectral_distances(fp: dict) -> pd.DataFrame:
    rows = [{"gen_a": a, "gen_b": b, "distance": float(np.sqrt(np.mean((fp[a] - fp[b]) ** 2)))}
            for a in fp for b in fp]
    return pd.DataFrame(rows)


def run_fourier(cfg: Config) -> pd.DataFrame:
    a = cfg.analysis.fourier
    out = analysis_dir(cfg, "fourier")
    manifest = load_manifest(cfg, a.variant)
    spectra, fp, prof_rows = {}, {}, []
    for g in generators(cfg):
        df = head_per_label(subset(manifest, g, "test"), a.n_images)
        if df.empty:
            continue
        real, fake = (mean_spectrum(cfg, df[df.label == l].path.tolist()) for l in (0, 1))
        spectra[g] = (real, fake)
        fp[g] = fingerprint(real, fake)
        for label, spec in (("real", real), ("fake", fake)):
            for r, v in enumerate(radial_profile(spec)):
                prof_rows.append({"generator": g, "label": label, "radius": r, "log_power": v})
        log.info("fourier: %s (%d images)", g, len(df))
    np.savez_compressed(out / "spectra.npz", **{f"{g}_{k}": s for g, (re, fa) in spectra.items()
                                                  for k, s in (("real", re), ("fake", fa))})
    pd.DataFrame(prof_rows).to_csv(out / "radial_profiles.csv", index=False)
    dist = spectral_distances(fp)
    dist.to_csv(out / "spectral_distance.csv", index=False)
    _plot(cfg, spectra, fp, dist, out)
    return dist


def load_spectral_distances(cfg: Config) -> dict:
    df = pd.read_csv(analysis_dir(cfg, "fourier") / "spectral_distance.csv")
    return {(r.gen_a, r.gen_b): r.distance for r in df.itertuples()}


def _plot(cfg: Config, spectra: dict, fp: dict, dist: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gens = list(spectra)
    fig, ax = plt.subplots(len(gens), 3, figsize=(7.5, 2.3 * len(gens)), squeeze=False)
    for i, g in enumerate(gens):
        real, fake = spectra[g]
        vmin, vmax = np.percentile(np.r_[real.ravel(), fake.ravel()], [1, 99.5])
        d = fake - real
        lim = np.abs(d).max()
        for j, (img, kw, name) in enumerate([(real, dict(vmin=vmin, vmax=vmax), "real"),
                                             (fake, dict(vmin=vmin, vmax=vmax), "fake"),
                                             (d, dict(vmin=-lim, vmax=lim, cmap="coolwarm"), "fake - real")]):
            ax[i, j].imshow(img, **kw)
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            ax[i, j].set_title(f"{title(cfg, g)}: {name}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for g, f in fp.items():
        ax[0].plot(f, label=title(cfg, g))
    ax[0].set(xlabel="spatial frequency (radius, px)", ylabel="log-power(fake) - log-power(real)",
              title="Radial fingerprint")
    ax[0].legend(fontsize=6), ax[0].grid(alpha=.3)
    m = dist.pivot(index="gen_a", columns="gen_b", values="distance").loc[gens, gens]
    im = ax[1].imshow(m.values, cmap="viridis")
    ax[1].set_xticks(range(len(gens)), [title(cfg, g) for g in gens], rotation=60, fontsize=6)
    ax[1].set_yticks(range(len(gens)), [title(cfg, g) for g in gens], fontsize=6)
    ax[1].set_title("Spectral distance between fingerprints")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out / "radial_fingerprints.png", dpi=120)
    plt.close(fig)
