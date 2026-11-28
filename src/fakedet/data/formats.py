"""JPEG helpers: encoding and estimating the quality factor of a JPEG file from its quantisation tables.

GenImage's reals are JPEG files while its fakes are PNGs, so file format alone separates the classes
(Grommelt et al.). The format-matched dataset keeps only reals stored with the standard libjpeg tables of
quality 96 and re-saves every fake at the same quality; `estimate_jpeg_quality` is how we recognise those reals.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

NOT_JPEG = -1       # `quality` of non-JPEG files
NONSTANDARD = 0     # `quality` of JPEG files whose tables are not a libjpeg quality setting


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


def recompress(img: Image.Image, quality: int) -> Image.Image:
    """Encode `img` as JPEG at `quality` and decode it again."""
    with Image.open(io.BytesIO(encode_jpeg(img, quality))) as out:
        return out.convert("RGB")


@lru_cache(maxsize=1)
def _reference_tables() -> tuple[np.ndarray, np.ndarray]:
    """(luma, chroma) quantisation tables of PIL/libjpeg for quality 1..100, each of shape (100, 64)."""
    luma, chroma = [], []
    for q in range(1, 101):
        with Image.open(io.BytesIO(encode_jpeg(Image.new("RGB", (8, 8)), q))) as im:
            luma.append(im.quantization[0])
            chroma.append(im.quantization[1])
    return np.asarray(luma, dtype=np.float64), np.asarray(chroma, dtype=np.float64)


def estimate_jpeg_quality(im: Image.Image) -> tuple[int, float]:
    """(quality, error) of the libjpeg quality setting whose tables are closest to the file's tables.

    error is the mean absolute difference of the quantisation tables; 0.0 means the file was written with
    exactly the standard tables of that quality. Returns (-1, inf) for anything that is not a JPEG.
    """
    tables = getattr(im, "quantization", None)
    if im.format != "JPEG" or not tables or 0 not in tables:
        return NOT_JPEG, float("inf")
    ref_luma, ref_chroma = _reference_tables()
    err = np.abs(ref_luma - np.asarray(tables[0], dtype=np.float64)).mean(1)
    if 1 in tables:
        err = (err + np.abs(ref_chroma - np.asarray(tables[1], dtype=np.float64)).mean(1)) / 2
    best = int(np.argmin(err))
    return best + 1, float(err[best])


def is_jpeg_quality(im: Image.Image, quality: int) -> bool:
    q, err = estimate_jpeg_quality(im)
    return q == quality and err == 0.0
