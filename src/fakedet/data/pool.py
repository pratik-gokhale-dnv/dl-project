"""The image pool: original image bytes of every (generator, split), plus a CSV describing them.

    <data_dir>/pool/<generator>/<split>/<real|fake>_<n>.<jpg|png>    the original bytes, never re-encoded
    <data_dir>/pool/<generator>__<split>.csv                         one row per image (see COLUMNS)
    <data_dir>/pool/<generator>__<split>.json                        bookkeeping that makes collection resumable

The raw and format-matched datasets (`build.py`) are both derived from the pool.
"""
from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from fakedet.data.formats import NONSTANDARD, NOT_JPEG, estimate_jpeg_quality
from fakedet.utils import ensure_dir, load_json, save_json

COLUMNS = ["path", "label", "cls", "fmt", "quality", "width", "height", "bytes", "src"]
EXT = {"JPEG": "jpg", "PNG": "png"}


def pool_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "pool"


def pool_csv(data_dir: str | Path, gen: str, split: str) -> Path:
    return pool_dir(data_dir) / f"{gen}__{split}.csv"


def state_path(data_dir: str | Path, gen: str, split: str) -> Path:
    return pool_dir(data_dir) / f"{gen}__{split}.json"


def pool_exists(data_dir: str | Path, gen: str, split: str) -> bool:
    return pool_csv(data_dir, gen, split).exists()


def load_pool(data_dir: str | Path, gen: str, split: str) -> pd.DataFrame:
    p = pool_csv(data_dir, gen, split)
    if not p.exists():
        raise FileNotFoundError(f"no image pool for {gen}/{split} in {data_dir} - run `fakedet data` first")
    return pd.read_csv(p, dtype={"cls": str, "src": str}, keep_default_na=False)


def load_state(data_dir: str | Path, gen: str, split: str) -> dict:
    p = state_path(data_dir, gen, split)
    return load_json(p) if p.exists() else {"shards_done": []}


def save_pool(data_dir: str | Path, gen: str, split: str, rows: list[dict], state: dict | None = None) -> None:
    """Atomically (over)write the pool CSV, and the bookkeeping file when given."""
    p = pool_csv(data_dir, gen, split)
    ensure_dir(p.parent)
    tmp = p.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, p)
    if state is not None:
        save_json(state, state_path(data_dir, gen, split))


def clear_pool(data_dir: str | Path, gen: str, split: str) -> None:
    shutil.rmtree(pool_dir(data_dir) / gen / split, ignore_errors=True)
    pool_csv(data_dir, gen, split).unlink(missing_ok=True)
    state_path(data_dir, gen, split).unlink(missing_ok=True)


def describe_image(raw: bytes) -> dict | None:
    """Format, JPEG quality and size of an encoded image, or None when it cannot be decoded."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            fmt = im.format
            if fmt == "JPEG":
                q, err = estimate_jpeg_quality(im)
                quality = q if err == 0.0 else NONSTANDARD
            else:
                quality = NOT_JPEG
            return {"fmt": fmt, "quality": quality, "width": im.width, "height": im.height}
    except Exception:
        return None


def add_image(data_dir: str | Path, gen: str, split: str, index: int, label: int, raw: bytes, cls: str | None,
              src: str = "") -> dict | None:
    """Write `raw` into the pool and return its row, or None if it is not a decodable JPEG/PNG."""
    info = describe_image(raw)
    if info is None or info["fmt"] not in EXT:
        return None
    rel = f"pool/{gen}/{split}/{'fake' if label else 'real'}_{index:06d}.{EXT[info['fmt']]}"
    dst = Path(data_dir) / rel
    ensure_dir(dst.parent)
    dst.write_bytes(raw)
    return {"path": rel, "label": int(label), "cls": "" if cls is None else str(cls), **info,
            "bytes": len(raw), "src": src}
