"""Sampling GenImage from the Hugging Face mirror `nebula/GenImage-arrow` into the image pool.

Mirror layout (assumed from public loaders of the dataset):  data/<split>/<generator folder>/data-*.arrow
Every shard is an Arrow IPC stream with the columns `image` (encoded bytes), `image_path`
(e.g. `ADM/train/nature/n01440764_10026.JPEG` or `ADM/train/ai/042_adm_00001.PNG`) and `label` (0 real, 1 fake).
Rows are ordered by label, so shards are visited in a spread-out order that mixes reals and fakes and
samples many ImageNet classes, and `data.max_shards` caps how many are downloaded per (generator, split).

Collection is resumable: after every shard the pool CSV and the list of finished shards are checkpointed.
"""
from __future__ import annotations

import heapq
import math
import random
import re
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow as pa

from fakedet.config import Config
from fakedet.data import pool
from fakedet.utils import log

_REAL_WNID = re.compile(r"^(n\d{8})_")      # ImageNet train reals: n01440764_10026.JPEG
_FAKE_CLASS = re.compile(r"^(\d+)_")        # generated fakes: <class index>_<generator>_<n>.PNG


# ----------------------------------------------------------------- naming / balancing
def class_key(path: str) -> str | None:
    """ImageNet class encoded in the file name, or None (e.g. ILSVRC2012_val_*.JPEG reals carry no class)."""
    name = PurePosixPath(path).name
    m = _REAL_WNID.match(name) or _FAKE_CLASS.match(name)
    return m.group(1) if m else None


def label_of(path: str, default: int) -> int:
    """0 = real (`nature` folder), 1 = fake (`ai` folder); `default` (the label column) if the path says nothing."""
    folders = PurePosixPath(path).parts[:-1]
    if "nature" in folders:
        return 0
    if "ai" in folders:
        return 1
    return int(default)


def balanced_pick(keys: list, n: int, rng: random.Random, counts: dict | None = None) -> list[int]:
    """Indices of `n` items chosen so that every key is picked about equally often.

    Water-filling: always take from the key with the fewest picks so far, where `counts` are picks made
    before this call (e.g. from earlier shards), so classes already well represented are served last.
    """
    buckets: dict = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    names = list(buckets)
    for b in buckets.values():
        rng.shuffle(b)
    counts = counts or {}
    heap = [(counts.get(k, 0), rng.random(), j) for j, k in enumerate(names)]
    heapq.heapify(heap)
    picked: list[int] = []
    while heap and len(picked) < n:
        c, _, j = heapq.heappop(heap)
        bucket = buckets[names[j]]
        picked.append(bucket.pop())
        if bucket:
            heapq.heappush(heap, (c + 1, rng.random(), j))
    return picked


def spread_order(n: int) -> list[int]:
    """0..n-1 ordered so that every prefix is spread evenly over the range (bit-reversal order)."""
    bits = max(1, (n - 1).bit_length())
    return sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))


# ----------------------------------------------------------------- Hugging Face access
def list_shards(cfg: Config, gen: str, split: str) -> list[str]:
    """Repo paths of the Arrow shards of one generator/split folder, sorted by name."""
    from huggingface_hub import HfApi
    d = cfg.data
    prefix = f"data/{d.hf_splits[split]}/{d.generators[gen].hf_dir}/"
    files = HfApi().list_repo_files(d.hf_repo, repo_type="dataset", revision=d.hf_revision)
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".arrow"))
    if not shards:
        raise RuntimeError(f"no .arrow shards under {prefix!r} in {d.hf_repo}; check data.hf_repo, "
                           f"data.hf_splits and data.generators.{gen}.hf_dir in the config")
    return shards


def download_shard(repo: str, filename: str, cache_dir: str | Path, revision: str | None = None,
                   attempts: int = 3) -> Path:
    """Download one shard into <cache_dir>/shards and return its local path."""
    from huggingface_hub import hf_hub_download
    for attempt in range(1, attempts + 1):
        try:
            return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                        local_dir=Path(cache_dir) / "shards"))
        except Exception as e:
            if attempt == attempts:
                raise
            log.warning("download of %s failed (%s), retrying (%d/%d)", filename, e, attempt, attempts)
            time.sleep(5 * attempt)


def open_shard(path: str | Path) -> pa.Table:
    try:
        return pa.ipc.open_stream(pa.memory_map(str(path))).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_file(pa.memory_map(str(path))).read_all()


def _image_bytes(column: pa.ChunkedArray) -> list[bytes]:
    if pa.types.is_struct(column.type):  # Hugging Face `Image` feature: {bytes, path}
        import pyarrow.compute as pc
        column = pc.struct_field(column, "bytes")
    return column.to_pylist()


# ----------------------------------------------------------------- collection
def _quota(target: int, n_shards: int) -> int:
    """Most images of one label taken from a single shard, so that every visited shard contributes:
    the target spread over the shards that can hold that label (about half of them), with 50% slack."""
    return max(1, math.ceil(1.5 * target / max(1, n_shards // 2)))


def _extract_shard(cfg: Config, gen: str, split: str, shard: str, table: pa.Table, need: dict, quota: dict,
                   counts: dict, rows: list[dict]) -> dict:
    """Move the selected images of one shard into the pool; returns {label: images added}."""
    d = cfg.data
    n = table.num_rows
    paths = table["image_path"].to_pylist() if "image_path" in table.column_names else [""] * n
    given = table["label"].to_pylist() if "label" in table.column_names else [-1] * n
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, (p, lab) in enumerate(zip(paths, given)):
        lab = label_of(p or "", -1 if lab is None else lab)
        if lab in by_label:
            by_label[lab].append(i)

    rng = random.Random(f"{d.seed}|{gen}|{split}|{shard}")
    chosen: list[tuple[int, int]] = []
    for lab, idx in by_label.items():
        take = min(need[lab], quota[lab], len(idx))
        if take <= 0:
            continue
        if d.class_balance:
            keys = [class_key(paths[i]) or "" for i in idx]
            pick = [idx[j] for j in balanced_pick(keys, take, rng, counts[lab])]
        else:
            pick = rng.sample(idx, take)
        chosen += [(lab, i) for i in pick]

    added = {0: 0, 1: 0}
    if not chosen:
        return added
    images = _image_bytes(table.take(pa.array([i for _, i in chosen]))["image"])
    for (lab, i), raw in zip(chosen, images):
        cls = class_key(paths[i]) or ""
        row = pool.add_image(cfg.project.paths["data_dir"], gen, split, len(rows), lab, raw, cls, src=paths[i])
        if row is None:  # undecodable image: the next shard makes up for it
            continue
        rows.append(row)
        counts[lab][cls] += 1
        added[lab] += 1
    return added


def collect_pool(cfg: Config, gen: str, split: str, n_real: int, n_fake: int, force: bool = False) -> pd.DataFrame:
    """Download shards of `gen`/`split` until the pool holds `n_real` reals and `n_fake` fakes
    (class-balanced where the file names allow it), or until `data.max_shards` shards were fetched."""
    d = cfg.data
    data_dir = cfg.project.paths["data_dir"]
    if force:
        pool.clear_pool(data_dir, gen, split)
    rows = pool.load_pool(data_dir, gen, split).to_dict("records") if pool.pool_exists(data_dir, gen, split) else []
    state = pool.load_state(data_dir, gen, split)
    done: list[str] = list(state["shards_done"])
    target = {0: n_real, 1: n_fake}
    counts = {lab: Counter(r["cls"] for r in rows if r["label"] == lab) for lab in (0, 1)}

    def need() -> dict:
        return {lab: target[lab] - sum(counts[lab].values()) for lab in (0, 1)}

    if max(need().values()) > 0:
        shards = list_shards(cfg, gen, split)
        order = [shards[i] for i in spread_order(len(shards))]
        budget = min(len(shards), d.max_shards) if d.max_shards else len(shards)
        quota = {lab: _quota(target[lab], budget) for lab in (0, 1)}
        for shard in order:
            if max(need().values()) <= 0:
                break
            if shard in done:
                continue
            if len(done) >= budget:
                log.warning("%s/%s: stopped after data.max_shards=%d shards (still missing %d real, %d fake)",
                            gen, split, budget, max(0, need()[0]), max(0, need()[1]))
                break
            local = download_shard(d.hf_repo, shard, cfg.project.paths["cache_dir"], d.hf_revision)
            added = _extract_shard(cfg, gen, split, shard, open_shard(local), need(), quota, counts, rows)
            done.append(shard)
            pool.save_pool(data_dir, gen, split, rows, {"shards_done": done, "target": [n_real, n_fake]})
            if not d.keep_shards:
                Path(local).unlink(missing_ok=True)
            log.info("%s/%s shard %d/%d: +%d real +%d fake (have %d/%d real, %d/%d fake)", gen, split, len(done),
                     budget, added[0], added[1], sum(counts[0].values()), n_real, sum(counts[1].values()), n_fake)
    if not rows:
        raise RuntimeError(f"collected no images for {gen}/{split}")
    pool.save_pool(data_dir, gen, split, rows, {"shards_done": done, "target": [n_real, n_fake]})
    log.info("%s/%s pool: %d real (%d classes), %d fake (%d classes)", gen, split, sum(counts[0].values()),
             len(counts[0]), sum(counts[1].values()), len(counts[1]))
    return pd.DataFrame(rows)
