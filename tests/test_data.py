import io
import random

import numpy as np
import pandas as pd
import pyarrow as pa
from PIL import Image

from fakedet.config import apply_overrides, planned_runs
from fakedet.data import genimage
from fakedet.data.build import build_datasets, load_manifest, subset
from fakedet.data.dataset import ImageDataset, fit_and_crop
from fakedet.data.formats import encode_jpeg, estimate_jpeg_quality, is_jpeg_quality


def _img(size=64, seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((size, size, 3)) * 255).astype(np.uint8))


def test_jpeg_quality_estimation():
    for q in (50, 75, 96, 100):
        im = Image.open(io.BytesIO(encode_jpeg(_img(), q)))
        assert estimate_jpeg_quality(im) == (q, 0.0)
    assert is_jpeg_quality(Image.open(io.BytesIO(encode_jpeg(_img(), 96))), 96)
    assert not is_jpeg_quality(Image.open(io.BytesIO(encode_jpeg(_img(), 75))), 96)
    buf = io.BytesIO()
    _img().save(buf, "PNG")
    assert estimate_jpeg_quality(Image.open(io.BytesIO(buf.getvalue())))[0] == -1


def test_class_key_and_label():
    assert genimage.class_key("ADM/train/nature/n01440764_10026.JPEG") == "n01440764"
    assert genimage.class_key("ADM/train/ai/042_adm_00001.PNG") == "042"
    assert genimage.class_key("BigGAN/test/nature/ILSVRC2012_val_00000009.JPEG") is None
    assert genimage.label_of("ADM/train/nature/a.JPEG", 1) == 0
    assert genimage.label_of("ADM/train/ai/a.PNG", 0) == 1
    assert genimage.label_of("a.png", 1) == 1


def test_balanced_pick_is_balanced():
    keys = ["a"] * 50 + ["b"] * 5 + ["c"] * 5
    picked = genimage.balanced_pick(keys, 12, random.Random(0))
    counts = pd.Series([keys[i] for i in picked]).value_counts()
    assert len(picked) == 12 and counts.max() <= 4 and set(counts.index) == {"a", "b", "c"}


def test_config_overrides_and_extends(cfg):
    assert cfg.data.source == "synthetic" and cfg.data.crop_size == 48
    assert cfg.models.clip_mid_mlp.head == "mlp" and cfg.models.clip_mid_mlp.random_config.hidden_size == 32
    assert len(planned_runs(cfg)) == 9 * 2  # raw sd14 + matched sd14
    raw = {"a": {"b": 1}}
    assert apply_overrides(raw, ["a.b=3"]) == {"a": {"b": 3}}


def _fake_shard(path, n_real, n_fake, cls_prefix):
    paths, labels, images = [], [], []
    for i in range(n_real):
        buf = io.BytesIO()
        _img(32, i).save(buf, "JPEG", quality=96)
        paths.append(f"G/train/nature/n{i % 5:08d}_{i}.JPEG"); labels.append(0); images.append(buf.getvalue())
    for i in range(n_fake):
        buf = io.BytesIO()
        _img(32, 100 + i).save(buf, "PNG")
        paths.append(f"G/train/ai/{i % 5:03d}_g_{i}.PNG"); labels.append(1); images.append(buf.getvalue())
    table = pa.table({"image": pa.array(images, pa.binary()), "image_path": paths, "label": labels})
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)


def test_collect_pool_from_arrow_shards(cfg, tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    names = []
    for k in range(4):  # real-only, real-only, fake-only, fake-only (the mirror orders rows by label)
        p = shard_dir / f"data-0000{k}.arrow"
        _fake_shard(p, 20 if k < 2 else 0, 20 if k >= 2 else 0, "x")
        names.append(p.name)
    monkeypatch.setattr(genimage, "list_shards", lambda *a, **k: names)
    monkeypatch.setattr(genimage, "download_shard", lambda repo, fn, *a, **k: shard_dir / fn)
    cfg.data.keep_shards = True
    genimage.collect_pool(cfg, "sd14", "train", n_real=16, n_fake=10)
    from fakedet.data.pool import load_pool
    df = load_pool(cfg.project.paths["data_dir"], "sd14", "train")
    assert (df.label == 0).sum() == 16 and (df.label == 1).sum() == 10
    assert set(df[df.label == 0].fmt) == {"JPEG"} and set(df[df.label == 1].fmt) == {"PNG"}
    assert (df[df.label == 0].quality == 96).all()
    assert df.cls.nunique() > 3  # class balanced inside shards
    assert all((tmp_path / "data_dir" / p).exists() for p in df.path)


def test_build_raw_and_matched(cfg):
    build_datasets(cfg)
    raw, matched = load_manifest(cfg, "raw"), load_manifest(cfg, "matched")
    data_dir = cfg.project.paths["data_dir"]
    # raw: JPEG reals vs PNG fakes (the shortcut); only SD v1.4 for training
    tr = subset(raw, "sd14", "train")
    assert set(tr[tr.label == 0].fmt) == {"JPEG"} and set(tr[tr.label == 1].fmt) == {"PNG"}
    assert set(raw[raw.split != "test"].generator) == {"sd14"}
    # matched: everything JPEG, reals Q96 only, balanced classes, all generators
    assert set(matched.fmt) == {"JPEG"} and set(matched.quality) == {96}
    assert set(matched[matched.split != "test"].generator) == {"sd14"}
    assert set(matched[matched.split == "test"].generator) == set(cfg.data.generators)
    for (g, s), grp in matched.groupby(["generator", "split"]):
        assert (grp.label == 0).sum() == (grp.label == 1).sum()
    for p in matched.path.sample(5, random_state=0):
        with Image.open(f"{data_dir}/{p}") as im:
            assert estimate_jpeg_quality(im) == (96, 0.0)
    # 90/10 train/val split of the training generator
    s = subset(matched, "sd14", ["train", "val"])
    assert abs((s.split == "val").mean() - cfg.data.val_fraction) < 0.02
    build_datasets(cfg)  # idempotent
    assert len(load_manifest(cfg, "matched")) == len(matched)


def test_crop_never_resizes_large_but_upsamples_small():
    big, small = _img(300), _img(100)
    out = fit_and_crop(big, 224, random_crop=False)
    assert out.size == (224, 224)
    assert np.array_equal(np.asarray(out), np.asarray(big)[38:262, 38:262])  # pure crop: no interpolation
    assert fit_and_crop(small, 224, random_crop=True).size == (224, 224)


def test_dataset_item(cfg):
    build_datasets(cfg)
    m = subset(load_manifest(cfg, "matched"), "sd14", "train")
    ds = ImageDataset(m, cfg.project.paths["data_dir"], 64, train=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and str(x.dtype) == "torch.uint8" and y.item() in (0.0, 1.0)
    assert ImageDataset(m, cfg.project.paths["data_dir"], 64, jpeg_quality=50)[0][0].shape == (3, 64, 64)
