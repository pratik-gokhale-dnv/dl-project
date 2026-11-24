# fakedet — Beyond Detection: do AI image detectors generalize to unseen generators?

CSE 7643 project (team *Finding Pixels*). A controlled comparison of five AI-generated-image detectors on
[GenImage](https://github.com/GenImage-Dataset/GenImage), with the JPEG/PNG format shortcut removed, plus a failure
analysis and tests of four hypotheses. Everything is plain PyTorch and is driven by **one config file** (`config.yaml`).

| Group | Detector | Config name(s) |
|---|---|---|
| Trained from scratch | Small CNN | `cnn` |
| | NPR (Tan et al., CVPR'24) and an RGB-input ablation | `npr`, `npr_rgb` |
| ImageNet pre-trained, fine-tuned | ResNet-50, DeiT-S (timm), 3 seeds each | `resnet50`, `deit_s` |
| Frozen CLIP ViT-L/14 + trainable head (UnivFD) | final-layer / layer-16 features × linear / MLP head | `clip_final_linear`, `clip_final_mlp`, `clip_mid_linear`, `clip_mid_mlp` |

> **Status / honesty note.** The whole pipeline (data → tune → train → eval → analyze → report) and the unit tests were run
> end-to-end on **synthetic data with tiny randomly-initialised models** on a CPU-only machine without Hugging Face access.
> The real-data path (GenImage download, pretrained timm / CLIP weights, GPU training) is implemented but **has not been
> executed**; the GenImage mirror layout was inferred from public loaders of the dataset. Expect to fix small things on the first
> real run, and start with `fakedet plan` and a reduced budget (see [Real run](#7-real-run)).

## 1. Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync                      # creates .venv and installs everything from pyproject.toml / uv.lock
uv run pytest                # optional: unit tests (a few minutes on CPU)
uv run fakedet --help
```

* `uv sync` pulls the default PyTorch wheel for your platform (on Linux that is the CUDA build, ~6 GB venv). On a CPU-only machine
  you can install a CPU build instead, e.g. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
* Real runs need internet access to `huggingface.co`. If your network requires auth, `export HF_TOKEN=...`.
* Disk: each GenImage generator folder is large. `data.keep_shards: false` deletes every `.arrow` shard after its images are
  extracted, and `data.max_shards` caps how many shards are fetched per (generator, split).

## 2. Quick start

```bash
# ~10 minute end-to-end check on CPU: synthetic data, tiny random models, every stage runs
uv run fakedet all --config config_smoke.yaml
less results_smoke/report.md

# The real experiment (GPU strongly recommended): edit config.yaml, then
uv run fakedet plan          # list every planned training run
uv run fakedet all           # data -> train -> eval -> analyze -> report
```

Each stage is **resumable**: finished work (downloaded shards, trained runs, predictions, cached CLIP features) is skipped on a re-run
unless you pass `--force`.

## 3. The single config file

`config.yaml` is the only thing you edit. Sections:

| Section | Controls |
|---|---|
| `project` | seed, device (`auto/cpu/cuda`), dataloader workers, output folders (`data_dir`, `runs_dir`, `results_dir`, `cache_dir`) |
| `data` | HF repo/revision, generators and their role (`train` = used for training+testing, `test` = unseen), sample counts, class balancing, `max_shards` budget, JPEG quality (96) for the format-matched version, which generators train on raw vs matched data |
| `training` | default epochs, batch size, optimiser, lr, weight decay, warm-up, AMP, early stopping, augmentation |
| `tuning` | lr × weight-decay grid searched on the validation split |
| `models` | the detectors (architecture, per-model overrides under `training:`, seeds) |
| `pipeline.stages` | what `fakedet all` runs, in order |
| `experiments` | which models to train and the default seeds |
| `eval` | batch size, bootstrap samples, CI level |
| `analysis` | settings of the Fourier / Grad-CAM / feature-distance / JPEG analyses and the hypothesis tests |

Overrides without editing the file (the key must already exist in the config):

```bash
uv run fakedet train --set training.epochs=2 --set project.device=cpu
uv run fakedet all --config my_config.yaml
```

A config may start with `extends: config.yaml` and list only what differs — `config_smoke.yaml` is an example.

## 4. Pipeline stages and commands

```bash
uv run fakedet data                # 1. sample GenImage, build raw + format-matched datasets + manifests
uv run fakedet tune                # 2. (optional) grid-search lr / weight decay per model on validation loss
uv run fakedet train               # 3. train every planned run (model × training generator × variant × seed)
uv run fakedet eval                # 4. cross-generator evaluation with 95% bootstrap CIs
uv run fakedet analyze             # 5. Fourier, Grad-CAM, feature distance, JPEG robustness, hypotheses H1-H4
uv run fakedet report              # 6. assemble results/report.md and figures
uv run fakedet all                 # the stages listed in pipeline.stages
uv run fakedet plan                # list planned runs and which are already trained
```

Filters (all optional): `--models cnn npr`, `--generators sd14`, `--variants matched`, `--only fourier jpeg` (analyze), `--force`.

```bash
uv run fakedet train --models resnet50 --variants matched --generators sd14
uv run fakedet eval --models cnn npr --force
uv run fakedet analyze --only gradcam hypotheses
```

To search hyper-parameters first, add `tune` before `train` in `pipeline.stages` (or run `fakedet tune`); with
`training.use_tuned: true` the best lr / weight decay per model (`runs/tuning/best_hparams.json`) is used by `train`.

## 5. What the pipeline does

### Data (`src/fakedet/data/`)
* **Source**: the Hugging Face mirror `nebula/GenImage-arrow`. `genimage.py` samples shards, keeps ImageNet-class-balanced
  reals/fakes per generator and split, and stores the original bytes in an on-disk *pool* (`pool.py`). The mirror's `test` folder is
  the benchmark's validation split.
* **Two dataset versions** (`build.py`, `formats.py`):
  * `raw` — as released (JPEG reals vs PNG fakes, so file format is a shortcut).
  * `matched` — format-matched after Grommelt et al.: only reals stored as JPEG quality 96 are kept and every fake is re-saved as
    JPEG-Q96; real and fake counts are equalised.
* `dataset.py`: crops (random for training, centre for evaluation) to `crop_size`=224 and never resizes, except images smaller than the
  crop (BigGAN, 128 px), which are upsampled first. BigGAN is therefore reported separately (resolution confound).
* `synthetic.py`: tiny synthetic generators (each with its own periodic fingerprint) so the full pipeline can run offline.
* Manifests (`<data_dir>/manifests/*.csv`) record file format, JPEG quality and size per image; `summary.csv` makes the format shortcut
  visible (`real_jpeg_frac` vs `fake_jpeg_frac`).

### Models (`src/fakedet/models/`)
Every model takes `[0,1]` float images, normalises internally, and returns one logit (fake vs real).
* `scratch.py` — `cnn` (plain conv stack) and `npr` (same network on the neighbouring-pixel residual `x − up(down(x))`; `input: rgb` gives the ablation).
* `pretrained.py` — timm `resnet50` and `deit_small_patch16_224`, fully fine-tuned with identical settings.
* `clip.py` — frozen CLIP ViT-L/14 (`transformers`) with a linear or 2-layer-MLP head on the projected embedding or the CLS token
  after block 16. Features are extracted once and cached (`cache/clip_features`), so only the head is trained.
* `__init__.py` — `build_model(cfg, name)`.

### Training (`src/fakedet/train.py`, `inference.py`)
`fit` = AdamW + warm-up/cosine schedule, AMP, gradient clipping, best checkpoint by validation loss, early stopping, CSV log and loss
curves. `train_run` is resumable; `tune` runs the grid; `inference.py` holds batched prediction, CLIP feature caching and embedding extraction.
Run names are `model__variant__generator__sSEED`, e.g. `resnet50__matched__sd14__s1`. Outputs are in `runs/<run>/`
(`best.pt`, `log.csv`, `curves.png`, `summary.json`).

### Evaluation (`src/fakedet/evaluate.py`)
Each run is tested on every generator, on both the raw and the matched test version. Metrics: accuracy, real/fake accuracy, AUC, AP.
95% CIs come from a **stratified percentile bootstrap** over test images; resamples are seeded per (variant, generator), so model and
condition comparisons are paired. Outputs: `results/per_run/*.csv`, `results/cross_generator.csv`, `results/summary.csv`
(`mean_acc_all`, `mean_acc_unseen`, per-generator accuracies with `_lo/_hi` and seed std), and per-image predictions in `results/preds/`.

### Analyses (`src/fakedet/analysis/`)
| Module | Output (`results/…`) | What |
|---|---|---|
| `fourier.py` | `fourier/` | mean log-spectra per generator, radial fingerprint (fake − real), spectral distance between generators |
| `gradcam.py` | `gradcam/` | Grad-CAM (own hook-based implementation, works for CNN and ViT) mean maps, example overlays, cosine similarity across generators |
| `features.py` | `features/` | distance of test-generator embeddings to the training generator's fake centroid vs accuracy, Spearman ρ |
| `jpeg.py` | `jpeg/` | accuracy after re-compressing test images at the configured JPEG qualities |
| `hypotheses.py` | `hypotheses.{md,json}` | tests of H1–H4 (below) |
| `report.py` | `report.md`, `figures/` | dataset stats, cross-generator tables with CIs, heatmaps, summary bar chart |

### Hypotheses
Verdicts are computed from the data with paired bootstrap CIs: *supported*, *partially supported*, *inconclusive* or *not supported*.

* **H1** — frozen CLIP features generalise to unseen generators better than the supervised detectors. Matched-trained, mean accuracy on unseen
  generators (test-only, excluding BigGAN): best CLIP variant vs best non-CLIP detector, with a paired CI on the difference.
* **H2** — format shortcut: raw-trained detectors lose accuracy when the *same* generator is tested on its format-matched version
  (drop ≥ `min_drop` with CI lower bound > 0); matched-trained models are the control.
* **H3** — NPR is more fragile to JPEG re-compression than the CNN: (NPR drop − CNN drop) > 0 with CI lower bound > 0.
* **H4** — spectral distance between training and test generators is negatively correlated with accuracy (per-model Spearman ρ over
  (train, test) generator pairs; supported if most models show significant negative ρ, partially if only some do).

## 6. Outputs

```
data/       pool/ (original images), matched/ (JPEG-Q96 versions), manifests/*.csv
runs/       <run name>/{best.pt,log.csv,curves.png,summary.json}, tuning/{best_hparams.json,grid_results.csv}
results/    report.md  summary.csv  cross_generator.csv  hypotheses.{md,json}
            per_run/  preds/  figures/  fourier/  gradcam/  features/  jpeg/
cache/      downloaded shards (if keep_shards) and cached CLIP features
```

All four folders are git-ignored and relocatable via `project.paths`.

## 7. Real run

1. Edit `config.yaml` (device, sample counts, `max_shards`, models, seeds, epochs). The defaults sample 5 000 train+val and 3 000 test images per
   label per generator. Full ImageNet-class balance (1 000 classes) would require nearly all shards of every generator (~100 GB each);
   `data.max_shards` is the budget — fewer shards means fewer ImageNet classes are covered. Pin `data.hf_revision` for reproducibility.
2. `uv run fakedet data`, then check `data/manifests/summary.csv`: in `raw` reals should be JPEG and fakes PNG; in `matched` both JPEG-Q96.
3. `uv run fakedet tune` (optional), `uv run fakedet train`, `uv run fakedet eval`, `uv run fakedet analyze`, `uv run fakedet report`
   — or simply `uv run fakedet all`.
4. Read `results/report.md` and `results/hypotheses.md`.

## 8. Repository layout

```
config.yaml           the single pipeline configuration
config_smoke.yaml     synthetic / tiny-model override used by the smoke test and unit tests
pyproject.toml, uv.lock
src/fakedet/
  cli.py              `fakedet` entry point (stages, filters, --set overrides)
  config.py           YAML loader (extends, overrides, validation, run planner)
  utils.py            logging, seeding, device selection, json/dir helpers
  data/               genimage.py, pool.py, formats.py, build.py, dataset.py, synthetic.py
  models/             base.py, scratch.py, pretrained.py, clip.py
  train.py            trainer, run resumption, hyper-parameter tuning
  inference.py        batched prediction, CLIP feature cache, embeddings
  evaluate.py         metrics, bootstrap CIs, cross-generator tables
  analysis/           fourier, gradcam, features, jpeg, hypotheses, report
tests/                pytest suite (data, models, training, metrics) run on synthetic data
```

## 9. Known limitations

* Real data and pretrained weights were not exercised in development (see the status note). Smoke-test numbers (e.g. CLIP at chance) are
  meaningless; they only show that every stage executes and the outputs are produced.
* The mirror's schema (`manifest.json`, `data/<split>/<generator folder>/*.arrow`, `image` / `image_path` / `label` columns) is an assumption;
  adjust `data.generators.*.hf_dir` / `data.hf_splits` in the config or `data/genimage.py` if it differs.
* Frozen-CLIP features are cached from centre crops only (no augmentation).
* Fourier / Grad-CAM / feature analyses use the reference run set in `analysis.reference` (SD v1.4, matched data, seed 0).
