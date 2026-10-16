# The denoiser's training toolkit

Trains Milestone 3's denoiser (docs/research/2026-10-14-denoiser-training-design.md)
on the denoiser-v1 data set, tests it once on the real marbles, and exports
its weights for the Metal pass. The record of the run, its curve and its
results: docs/research/2026-10-15-denoiser-training.md.

Its own environment, outside the C++ build: nothing the renderer builds or
runs depends on it.

## The environment

- Python **3.12.15** (`.python-version`; `requires-python = "==3.12.*"`).
- **uv** 0.13 resolves and installs; `uv.lock` pins every package, direct
  and transitive, by exact version and SHA-256. Direct dependencies:
  torch 2.14.1 (the M3 Max's GPU through its `mps` backend), numpy 2.5.4,
  pillow 12.3.0.
- PFM images are read directly (`pfm.py`), no image library.
- For the evaluation only: `ffmpeg` (the MP4) and, optionally, Open Image
  Denoise 2.5.1's `oidnDenoise` (`brew install open-image-denoise`), as a
  comparison.

```sh
cd training
uv sync --locked                       # exactly uv.lock; refuses a stale lock
uv run --locked python -m unittest test_training -v
```

## The data

The data set is read, never written, and every file is checked against
its manifest's SHA-256 as it is loaded. Check it whole first (needs a
configured build in the checkout, as the script reads its CMake cache):

```sh
python3 scripts/make_denoiser_dataset.py --verify --out ../serenity-data/denoiser-v1
```

Training reads the train and validation splits only. The test split is read
by `evaluate.py` alone.

The test sequences' warm-up frames (the 8 input frames before each, so the
recurrent network is scored with history, as it runs) are rendered once by
the data set's own renderer binary, checked by its SHA-256:

```sh
uv run --locked python render_warmup.py --data ../../serenity-data/denoiser-v1 \
    --checkout .. --out ../../serenity-data/denoiser-v1-warmup
```

## Training

`config.json` holds the run's settings; paths in it are relative to it.

```sh
# OIDN's learning-rate finder: an exponential sweep, its curve in find_lr.csv
uv run --locked python train.py find-lr --config config.json --out ../../serenity-data/denoiser-v1-runs/find-lr

# the run: log.csv every epoch, best.pt at the best validation loss, last.pt to resume
uv run --locked python train.py train --config config.json --out ../../serenity-data/denoiser-v1-runs/v1
```

`train` resumes from `last.pt` if the run directory has one, and refuses a
directory holding another config's run. Each checkpoint records the config,
the data set manifest's SHA-256, the git commit and whether `training/` had
uncommitted changes.

## The test, once

```sh
uv run --locked python evaluate.py --checkpoint ../../serenity-data/denoiser-v1-runs/v1/best.pt \
    --data ../../serenity-data/denoiser-v1 --warmup ../../serenity-data/denoiser-v1-warmup \
    --references ../.cache/references --oidn /opt/homebrew/bin/oidnDenoise \
    --out ../../serenity-data/denoiser-v1-runs/v1/test --images IMAGES_DIR
```

It refuses an `--out` that already holds results. Tables:
`uv run --locked python report.py wide=RUN_A/test/results.json small=RUN_B/test/results.json`.

`config.json` is the small network (max_lr 1e-3), `config-wide.json` the wide
one (max_lr 5e-3); docs/research/2026-10-15-denoiser-training.md has both runs.

## The export

```sh
uv run --locked python export.py --checkpoint ../../serenity-data/denoiser-v1-runs/v1/best.pt \
    --out ../../serenity-data/denoiser-v1-model
```

`weights.bin` (float32), `layout.json` (every tensor's place, the network's
structure, its inputs and outputs) and `SHA256SUMS`; the export reads the
weights back by the layout alone and checks they compute what the
checkpoint does, bit for bit.

## Files

| File | What |
|---|---|
| `pfm.py` | PFM, as the renderer writes it |
| `data.py` | the manifest, the input channels, demodulation, augmentation, crops and tiles |
| `model.py` | the network: U-Net features, upscaling kernels, the history blend |
| `loss.py` | PU, MS-SSIM, OIDN's l1_msssim, the temporal term, frame weights |
| `train.py` | the learning-rate finder and training |
| `regions.py` | the test's region masks, by a ray through each pixel's center |
| `tonemap.py` | the renderer's tone map, for displayed error |
| `render_warmup.py` | the test sequences' warm-up frames |
| `evaluate.py` | the test |
| `export.py` | the weights for Phase 4 |
| `report.py` | markdown tables from one or more test results |
| `benchmark.py` | a rough MPS time a frame |
| `test_training.py` | the conventions, on synthetic data |
