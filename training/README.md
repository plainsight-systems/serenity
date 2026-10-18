# The denoiser's training toolkit

Trains Milestone 3's denoiser (docs/research/2026-10-14-denoiser-training-design.md)
on the denoiser-v1 data set, tests it once on the real marbles, and exports
its weights for the Metal pass. The record of the run, its curve and its
results: docs/research/2026-10-15-denoiser-training.md.

Its own environment, outside the C++ build: nothing the renderer builds or
runs depends on it. Everything it reads and writes lives beside the
checkout, never in it, in `../serenity-data/` (from the checkout's root;
`../../serenity-data/` from this directory):

| Directory | What | Made by |
|---|---|---|
| `denoiser-v1/` | the data set, 32.3 GB, its manifest | `scripts/make_denoiser_dataset.py` |
| `denoiser-v1-warmup/` | the test sequences' warm-up frames | `render_warmup.py` |
| `denoiser-v1-runs/` | each run's log, checkpoints and test results | `train.py`, `evaluate.py` |
| `denoiser-v1-model/wide/` | the exported network the renderer reads | `export.py`, or `make weights` from the published copy |

To run the published network without training, `make weights` (from the
checkout's root) downloads it from the project's media host
(`https://media.plainsight-systems.com/denoiser-v1-model/`) and checks each
file against the SHA-256 pinned in `scripts/fetch_denoiser_weights.sh`.

The whole cycle, in order: render the data set, render the warm-up frames,
train, test once, export, check the renderer against PyTorch, run it.
Each step below.

## The environment

- Python **3.12.15** (`.python-version`; `requires-python = "==3.12.*"`).
- **uv** 0.13 resolves and installs; `uv.lock` pins every package, direct
  and transitive, by exact version and SHA-256. Direct dependencies:
  torch 2.14.1, numpy 2.5.4, pillow 12.3.0. Training runs on the RTX 3090
  through CUDA (the lock holds Linux's CUDA wheels), the split held in the
  host's memory and each batch's crops handed to the GPU (`config-wide.json`'s
  `hold`); evaluation and the benchmark run on the M3 Max through `mps`.
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

From the checkout's root, with the system Python (the script uses only the
standard library). It builds `build/native-release/serenity-headless` and
renders from a clean, committed tree, recording the commit and the
binary's SHA-256 in the manifest; it resumes where it stopped, and refuses
a manifest made by another commit, so a new render of the data set goes
into a new directory. denoiser-v1 took 4.0 h on the M3 Max
(docs/research/2026-10-14-denoiser-dataset.md):

```sh
python3 scripts/make_denoiser_dataset.py --print-plan                                   # the plan as JSON; renders nothing
python3 scripts/make_denoiser_dataset.py --estimate --out ../serenity-data/denoiser-v1  # the first input and target, timed
python3 scripts/make_denoiser_dataset.py --out ../serenity-data/denoiser-v1             # render it, or resume
```

Training reads the data set, never writes it, and checks every file against
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

A config holds a run's settings; paths in it are relative to it.
`config-wide.json` is the demo's network (543k parameters, max_lr 5e-3);
docs/research/2026-10-15-denoiser-training.md has its run (8.3 h on the
M3 Max, sharing the GPU with the other runs), and the smaller network
trained beside it, since removed.

```sh
# OIDN's learning-rate finder: an exponential sweep, its curve in find_lr.csv
uv run --locked python train.py find-lr --config config-wide.json --out ../../serenity-data/denoiser-v1-runs/find-lr

# the runs: log.csv every epoch, best.pt at the best validation loss, last.pt to resume
uv run --locked python train.py train --config config-wide.json --out ../../serenity-data/denoiser-v1-runs/v1-wide
```

`train` resumes from `last.pt` if the run directory has one, and refuses a
directory holding another config's run. Each checkpoint records the config,
the data set manifest's SHA-256, the git commit and whether `training/` had
uncommitted changes. `--data DIR` reads the data set from elsewhere than
the config's path.

## The test, once

The real marbles are the test, read by `evaluate.py` alone, once a run has
ended; the references are `make reference`'s at t = 8 and t = 120, in the
checkout's `.cache/references/`:

```sh
uv run --locked python evaluate.py --checkpoint ../../serenity-data/denoiser-v1-runs/v1-wide/best.pt \
    --data ../../serenity-data/denoiser-v1 --warmup ../../serenity-data/denoiser-v1-warmup \
    --references ../.cache/references --oidn /opt/homebrew/bin/oidnDenoise \
    --out ../../serenity-data/denoiser-v1-runs/v1-wide/test --images IMAGES_DIR
```

It refuses an `--out` that already holds results. `--dry-run-on-validation`
runs the same code on the validation split instead, to check the script
before the test. Its tables:

```sh
uv run --locked python report.py wide=../../serenity-data/denoiser-v1-runs/v1-wide/test/results.json
```

A rough MPS time a frame: `uv run --locked python benchmark.py --checkpoint RUN/best.pt`.

## The export

```sh
uv run --locked python export.py --checkpoint ../../serenity-data/denoiser-v1-runs/v1-wide/best.pt \
    --out ../../serenity-data/denoiser-v1-model/wide
```

`weights.bin` (float32), `layout.json` (every tensor's place, the network's
structure, its inputs and outputs) and `SHA256SUMS`; the export reads the
weights back by the layout alone and checks they compute what the
checkpoint does, bit for bit.

## The renderer against PyTorch

The renderer's denoise pass (src/metal/passes/denoise/) runs the exported
network on the GPU; `denoise_check.py` makes a fixture of renderer images
and PyTorch's outputs for it, and compares the GPU's outputs with them
(docs/research/2026-10-15-denoiser-live.md):

```sh
uv run --locked python denoise_check.py fixture --network ../../serenity-data/denoiser-v1-model/wide \
    --data ../../serenity-data/denoiser-v1 --sequence gen-0125 --frames 8 --out CHECK_DIR
../build/native-release/tests/gpu/denoise_check CHECK_DIR GPU_DIR
uv run --locked python denoise_check.py compare --fixture CHECK_DIR --gpu GPU_DIR
```

The test's own fixture, `tests/data/denoise`, was made by `fixture
--random-network --sequence gen-0121 --frames 3 --crop 512 14 40 22`; ctest
runs it on every build (`tests/gpu/denoise_test.cpp`).

## Running it

The denoised graph names the export by a path relative to `graphs/`, so
from the checkout's root:

```sh
make run GRAPH=graphs/restir_pt_denoised.toml SCENE=scenes/marbles.toml
```

A graph's `[denoise] weights` names another export.

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
| `denoise_check.py` | the renderer's denoise pass against PyTorch: fixtures and the comparison |
| `report.py` | markdown tables from one or more test results |
| `benchmark.py` | a rough MPS time a frame |
| `test_training.py` | the conventions, on synthetic data |
