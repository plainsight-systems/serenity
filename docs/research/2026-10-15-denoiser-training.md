# The denoiser's training: Phase 3

2026-10-15. Phase 3 of Milestone 3 (2026-10-14-denoiser-training-design.md,
"Phases"): the `training/` toolkit, the runs on denoiser-v1, their curves,
the test on the real marbles, and the weights exported for Phase 4. Apple
M3 Max (128 GB), PyTorch 2.14.1 on its `mps` backend.

In short:

- Two sizes trained, 300 epochs each: small (294k parameters, 20.5 G
  multiply-adds a frame) and wide (543k, 36.3 G, OIDN's `unet` scale). Best
  validation loss 0.03702 and 0.03654; no sign of overfitting.
- On the real marbles, after 8 warm-up frames, the wide network's displayed
  error is 0.29 to 0.39 of the raw input's over the whole frame and 18 to
  38% below Open Image Denoise's; the small one's 1 to 9% above the wide
  one's. Against the 65,536-sample references at t = 120: 0.00163 (input
  0.00462, OIDN 0.00221); the 4,096-sample targets' own error is a tenth of
  the network's, on glass and metal too.
- The table flickers: 36 to 47% more frame-to-frame change than the targets
  once every firefly is awake; the temporal term is under 3% of the loss.
  A decision for the user (queued).
- Both networks exported for Phase 4; which one runs waits for its Metal
  cost.

## A new dependency, declared

- **What:** `training/`, a Python environment of its own: Python 3.12.15
  (`.python-version`, `requires-python = "==3.12.*"`), resolved and installed
  by uv 0.13 from `uv.lock`, which pins every package, direct and
  transitive, by exact version and SHA-256 (`uv sync --locked` refuses a
  lock that no longer matches `pyproject.toml`). Direct: torch 2.14.1, numpy
  2.5.4, pillow 12.3.0 (the PNGs). PFM is read directly; no OpenImageIO.
  For the evaluation only, outside the lock: ffmpeg (the MP4) and Open Image
  Denoise 2.5.1's `oidnDenoise` from Homebrew (`brew install
  open-image-denoise`, Intel's release built by homebrew-core; chosen over
  the unofficial PyPI wheels `oidn` and `pyoidn`).
- **Why:** the design chose PyTorch for training (its reference code, Open
  Image Denoise's toolkit, is PyTorch).
- **What it affects:** nothing the renderer builds or runs. No CMake target,
  test or graph reads it; the C++ toolchain pin is untouched. Its risk
  surface is the training's own reproducibility (below) and the weights it
  exports, which Phase 4 reads as a plain file with a checksum.

## What was built

`training/` (its README has the commands):

| File | What |
|---|---|
| `data.py` | the manifest's train and validation splits, every file checked against its SHA-256 as read; the input channels; demodulation; OIDN's augmentation |
| `model.py` | the network |
| `loss.py` | OIDN's PU transfer and l1_msssim, the temporal term, the frame weights |
| `train.py` | OIDN's learning-rate finder, and training |
| `evaluate.py` | the test, once; a labeled dry run on validation |
| `regions.py`, `tonemap.py` | the test's region masks; the renderer's tone map |
| `render_warmup.py` | the test sequences' warm-up frames |
| `export.py`, `benchmark.py` | the weights for Phase 4; a rough MPS time |
| `test_training.py` | the conventions, 13 tests on synthetic data |

### The inputs

Per frame at 960 × 540, 12 channels (`data.py`):

- **Noisy radiance divided by the albedo factor.** NRD's rule for dividing
  by albedo (`Shaders/NRD.hlsli`, `NRD_MaterialFactors`: `lerp(0.02, 1,
  albedo)`, "smaller values may lead to instabilities"): factor = 0.02 +
  0.98 a per channel, so a black albedo divides by 0.02, never 0. The output
  is multiplied by the same factor of the full-size albedo. Open Image
  Denoise takes albedo as a guide only and does not demodulate; NRD's is the
  production rule for it.
- **Albedo**, clipped to [0, 1] as OIDN clips it.
- **Normal in view space** (right, up, toward the camera), from the
  renderer's world normal and the scene's camera: mirroring an image mirrors
  the scene across a plane of the camera, which negates one view-space
  component exactly; in world space it is no axis at all. Phase 4 needs the
  same 3 × 3 transform (queued).
- **Depth, normalized per frame:** log(1 + z / m), m the median view depth
  over the frame's surface pixels (z > 0); the sky's 0 stays 0. The median,
  not the mean or the maximum: generated scenes that show the distant ground
  have a maximum of 35 m against 0.45 m at the marbles.
- **Motion** dx, dy (NRD's IN_MV, "old = new + MV", input pixels); dz
  unused. The convention was checked on the data: moving last frame's
  full-size albedo by +MV matches this frame's where things move (mean error
  0.010 to 0.015 in the brass sphere scenes) better than by -MV (0.020 to
  0.070) or not at all (0.011 to 0.036).

The targets: the 4,096-sample radiance and the full-size albedo at 1920 ×
1080. The camera is still in every scene (motion comes from fireflies and the
brass sphere's flight alone), so the warm-up and history are about moving
lights, not a moving view.

### Augmentation

OIDN's (`training/dataset.py`, `TrainingDataset`), for sequences: one draw
for all 8 frames; a random 128 × 128 input crop and the aligned 256 × 256
target crop; horizontal flip, vertical flip, transpose, each with
probability 1/2, carried into the vectors (horizontal: normal right and dx
negated; vertical: normal up and dy negated; transpose: dx and dy swapped
and (right, up) becoming (-up, -right), as image rows run down and view up
runs up); a random color-channel permutation shared by input radiance,
input albedo, target radiance and target albedo; a random permutation of
the normal's axes after the flips; and with probability 0.01 the input and
target radiance zeroed (OIDN: it "prevents ghosting artifacts when the main
feature is entirely black", which here is the dark opening). Not taken:
OIDN's zero padding of 10% of crops, which teaches its network the padding
of images to its alignment; here the filter never reads outside the image.

An epoch is 32 random crops of each of the 39 training sequences (1,248
sequences, 78 steps of 16): about the training frames' area once.
Validation is OIDN's fixed tiling (`ValidationDataset`): 7 × 4 whole tiles
a frame, centered, no augmentation, 112 tile-sequences, the same loss.

### The network

`model.py`, per frame:

1. **History.** Last frame's output, kept before the albedo multiply at
   1920 × 1080, is moved by this frame's motion: output pixel X reads last
   frame's output at X + 2 MV(⌊X / 2⌋), bilinearly. The motion is upsampled
   by nearest neighbor, so a vector is one surface's, never a blend of two
   across an edge. A read outside the last frame is not history (`valid`
   0), nor is anything on a sequence's first frame.
2. **Features.** A U-Net at 960 × 540, OIDN's layout (3 × 3 convolutions
   and ReLU, 2 × 2 max pools, nearest upsampling, skips by concatenation,
   four levels below the input's), reading 28 channels: PU of the
   demodulated radiance, albedo, normal, depth, motion, and the moved
   history (PU) and its validity, each 2 × 2 block of output pixels folded
   into channels at its input pixel (pixel unshuffle, nothing averaged). A
   1 × 1 head gives each input pixel 4 × 31 scores, 31 for each of the four
   output pixels over it.
3. **Kernels, the upscaling.** Output pixel X's kernel is over the input
   pixels around p = ⌊X / 2⌋, the input pixel whose footprint holds X's
   center: three 3 × 3 sets at spacings 1, 2 and 4, each set's 9 scores
   softmax-normalized and used to average the noisy demodulated radiance,
   and a softmax over 3 more scores blending the three averages. Sub-pixel
   position: X's center is a quarter input pixel from p's center toward one
   corner; the four output pixels sharing p each get their own 31 scores, so
   each kernel leans toward its own quarter as it learns to; no bilinear
   weight is imposed. (Whether Thomas et al. 2022 handle the sub-pixel
   offset the same way could not be checked: their paper could not be
   fetched, below.) A tap outside the image has its score masked out
   before the softmax.
4. **Blend.** One more score, through a sigmoid and times `valid`: out =
   alpha × history + (1 - alpha) × filtered, in demodulated light. This is
   the next frame's history.
5. **Albedo.** The radiance is out × factor(full-size albedo).

So every output is a convex combination of noisy inputs and history: never
negative, never brighter than the brightest of them. In training it is
unrolled over the 8 frames, each frame fed its own previous output, the
gradient through all 8.

### The loss

`loss.py`, per frame t on P = PU(output), T = PU(target):

- **Spatial: OIDN's default** l1_msssim, 0.16 L1 + 0.84 (1 - MS-SSIM), with
  OIDN's MS-SSIM for its RT filter (5 scales weighted 0.2, an 11-tap
  Gaussian window of sigma 1.5). Written again from OIDN's definition and
  checked against its code: equal to the 7th digit (0.9558438658714294 both,
  on a random pair). OIDN's PU transfer (Aydin et al. 2008's PU2 fit,
  normalized so 65,504 maps to 1), identical to OIDN's at 8 test values.
  Without OIDN's autoexposure: the renderer has none (its exposure is fixed,
  core/passes/tone_map.h), so the loss weighs error at the brightness the
  viewer sees.
- **Temporal: Chaitanya et al. 2017** ("Interactive Reconstruction of Monte
  Carlo Image Sequences using a Recurrent Denoising Autoencoder", §4.4):
  L1 of the difference between consecutive outputs' and consecutive
  targets' frame differences, unwarped. Weight: theirs relative to the
  spatial term, w_s / w_t = 0.8 / 0.1, "to approximately equalize the
  scales" (their third term, a gradient-domain L1, is not in the design): so
  0.125.
- **Frames weighted as Chaitanya et al. weigh them:** a Gaussian rising to
  the last frame, "to amplify temporal gradients", (0.011, 0.044, 0.135,
  0.325, 0.607, 0.882, 1) for 7 frames, exp(-(7 - t)² / 8) for 8.
- Thomas et al. 2022's own loss weights could not be read (the paper's PDF
  is 121 MB, past what could be fetched); Chaitanya et al.'s are the
  published real-time weights used.

## The learning rate

OIDN's `find_lr` (Adam, the rate raised exponentially from 1e-8 to 0.1
over 400 steps, the loss smoothed with beta 0.98, stopped at 4x its best):
the smoothed loss is flat at 0.067 to 0.072 (batch-to-batch noise, the
crops' content) up to about 1.5e-4, falls to its minimum 0.0624 at 6.4e-3,
and diverges past 2e-2 (0.078 at 4.6e-2, 0.090 at 0.1). OIDN reads the
plot for where learning starts and where it diverges; the descent here is
too noisy to read a single point from, so three 15-epoch one-cycle trials
across it decided (validation loss at the end): max_lr 2e-4 0.0526, 1e-3
0.0484, 5e-3 0.0480. OIDN's own default for batch 16 would be 5e-5.

- **5e-3, the trials' best, did not hold for the small network over 300
  epochs:** at epoch 159 (rate 2.9e-3) its validation loss jumped 0.039 ->
  0.058, and 48 epochs later it was still 0.041 (its best had been 0.0389,
  at epoch 139). It was stopped
  at epoch 207 and run again at 1e-3, the trials' second; its record is
  kept (runs/v1-small). The wide network held at 5e-3 through all 300, its
  brief spikes at most 0.001 (0.0385 to 0.0396 around epoch 158), each gone
  within a few epochs.
- So the two networks are compared each at a rate it trains stably at,
  not at one rate: wide at 5e-3, small at 1e-3.

## The runs

Both 300 epochs (23,400 steps of 16 sequences of 8 frames), OIDN's
one-cycle (15% warm-up, max_lr / 25 to max_lr, cosine down to max_lr /
2.5e5), Adam, float32, seed 20261015, data manifest SHA-256 ae820e1d...,
code at c7b27dd (wide) and 34b091e (small), `training/` clean. The wide run
took 8.3 h, sharing the GPU with the other runs; the small rerun 5.7 h.

| | parameters | U-Net multiply-adds a frame | max_lr | best validation loss (epoch) |
|---|---|---|---|---|
| small: widths 24, 32, 48, 64, bottleneck 64 | 294,180 | 20.5 G | 1e-3 | **0.03702** (285) |
| wide: widths 32, 48, 64, 80, bottleneck 96 (OIDN's `unet` scale) | 542,748 | 36.3 G | 5e-3 | **0.03654** (285) |

The curve (validation; train is noisier, its crops random):

| epoch | wide: lr | train | validation | small: lr | train | validation |
|---|---|---|---|---|---|---|
| 1 | 2.0e-4 | 0.0628 | 0.0627 | 4.0e-5 | 0.0663 | 0.0694 |
| 10 | 6.8e-4 | 0.0526 | 0.0509 | 1.4e-4 | 0.0611 | 0.0639 |
| 25 | 2.9e-3 | 0.0481 | 0.0449 | 5.9e-4 | 0.0523 | 0.0492 |
| 50 | 5.0e-3 | 0.0451 | 0.0414 | 1.0e-3 | 0.0475 | 0.0433 |
| 100 | 4.5e-3 | 0.0412 | 0.0394 | 8.9e-4 | 0.0413 | 0.0398 |
| 150 | 3.2e-3 | 0.0407 | 0.0387 | 6.4e-4 | 0.0408 | 0.0386 |
| 200 | 1.7e-3 | 0.0388 | 0.0373 | 3.3e-4 | 0.0384 | 0.0376 |
| 250 | 4.6e-4 | 0.0389 | 0.0368 | 9.2e-5 | 0.0390 | 0.0371 |
| 285 | 4.3e-5 | 0.0387 | **0.03654** | 8.5e-6 | 0.0388 | **0.03702** |
| 300 | 2e-8 | 0.0381 | 0.03657 | 4e-9 | 0.0383 | 0.03703 |

- **It stopped improving where the schedule ends:** the last 15 epochs of
  each are flat to the fifth digit as the rate anneals to nothing. Whether
  a longer cycle would go lower is not known: the validation loss was still
  falling by about 0.0005 per 50 epochs through the anneal (queued).
- **The gap:** validation is below training throughout, mean -0.0020
  (wide) and -0.0016 (small) over the last 30 epochs. The training loss is
  on random, augmented crops (1% of them zeroed, color and normal axes
  shuffled, many crops in the dark corners of opening frames) and the
  validation loss on fixed whole tiles, so the levels are not comparable
  one to one; what matters is the trend, and nothing in either curve turns
  up while training keeps falling: no sign of memorizing the 39 sequences
  within 300 epochs.
- **The temporal term is small in the loss:** its raw value settles at
  0.0083 against the spatial term's 0.036, so at Chaitanya et al.'s 0.125
  it is under 3% of the loss. Its consequence is in the flicker results.
- **Reproducibility:** two 2-epoch runs from the same seed and code draw
  the same crops but are not bit-identical on MPS: their losses agree to
  four figures (validation 0.058223 against 0.058211 after 156 steps).
  PyTorch makes no bit-exactness promise for MPS; a rerun of a whole run
  will land near, not on, these numbers.

## How fast, roughly

`benchmark.py`, the GPU otherwise idle, batch 1, 960 x 540 (padded to 544)
to 1920 x 1080 with history, median of 50 frames:

| | float32 whole frame | float32 U-Net | float16 whole frame | float16 U-Net |
|---|---|---|---|---|
| small | 32.1 ms | 9.2 ms | 25.0 ms | 7.2 ms |
| wide | 35.2 ms | 13.2 ms | 27.6 ms | 9.6 ms |

Most of the whole frame is not the U-Net: it is PyTorch's kernels for the
taps (unfold, three softmaxes and weighted sums at 4 sub-pixels), the
history's warp and the pixel shuffles, each a separate dispatch with its
tensors written out between. In one Metal kernel those are a gather of 27
taps per output pixel. Phase 4's measurement is the one that counts; this
says the U-Net, the part that scales with the widths, is 7 to 10 ms in
PyTorch's half precision, and that the wide network costs 1.3 to 1.4x the
small one there.

## The test, once

`evaluate.py`, after both runs ended, on the real marbles (never trained
on), each network's best-validation checkpoint, one run each. The wide
network was tested first, at the user's request, as soon as its run ended;
the small one when its run ended. Neither result was used to choose or
change anything.

- **History, as at runtime:** each test sequence is preceded by its 8
  input frames rendered before it (`render_warmup.py`, the data set's own
  binary by SHA-256; the 8 shared frames checked against the manifest:
  features byte for byte; radiance within float rounding, below). The
  network runs the 16 and the 8 that are the data set's are scored. Without
  this the references' frames, t = 8 and t = 120, which are each
  sequence's first frame, would be scored on a cold start.
- **Displayed MSE:** both images through the renderer's tone map
  (`tonemap.py`: exposure 0, bloom 0.04, PBR Neutral, sRGB, 8 bits), MSE of
  the 8-bit values over 255, averaged over the region's pixels and
  channels, as Milestone 2 measured it. `tonemap.py` against the renderer's
  own PNG of the same 16-sample frame (t = 120): 0.8% of values off by one
  step, none by more; MSE between them 1.2e-7, under a thousandth of the
  smallest figure below.
- **Regions:** `regions.py`, a pinhole ray through each pixel's center
  against the scene's spheres and boxes, the first hit's material: glass and
  metal 208,797 pixels, porcelain 81,408, table 1,783,395, against
  Milestone 2's 209k, 81k and 1.77M.
- **Compared with:** the raw ReSTIR PT + MNEE input upscaled bilinearly;
  Open Image Denoise 2.5.1 (`oidnDenoise -f RT -q high`, HDR color, albedo
  and normal, each frame alone at 960 x 540, then bilinearly upscaled: it
  does not upscale; a single-frame quality yardstick, as the design has it).

#### Displayed MSE against the 4,096-sample targets, mean of each sequence's 8 frames

**whole**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | wide / input |
|---|---|---|---|---|---|
| marbles-t002 | 0.00009 | 0.00008 | 0.00024 | 0.00012 | 0.36 |
| marbles-t004 | 0.00070 | 0.00075 | 0.00188 | 0.00106 | 0.38 |
| marbles-t006 | 0.00160 | 0.00163 | 0.00452 | 0.00248 | 0.36 |
| marbles-t008 | 0.00197 | 0.00207 | 0.00688 | 0.00318 | 0.29 |
| marbles-t030 | 0.00181 | 0.00186 | 0.00465 | 0.00222 | 0.39 |
| marbles-t060 | 0.00165 | 0.00171 | 0.00443 | 0.00217 | 0.37 |
| marbles-t120 | 0.00170 | 0.00175 | 0.00464 | 0.00224 | 0.37 |

**glass+metal**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | wide / input |
|---|---|---|---|---|---|
| marbles-t002 | 0.00044 | 0.00041 | 0.00090 | 0.00058 | 0.49 |
| marbles-t004 | 0.00437 | 0.00478 | 0.00987 | 0.00694 | 0.44 |
| marbles-t006 | 0.00952 | 0.00968 | 0.0249 | 0.0160 | 0.38 |
| marbles-t008 | 0.0118 | 0.0124 | 0.0394 | 0.0201 | 0.30 |
| marbles-t030 | 0.0106 | 0.0109 | 0.0291 | 0.0149 | 0.36 |
| marbles-t060 | 0.00920 | 0.00962 | 0.0253 | 0.0144 | 0.36 |
| marbles-t120 | 0.00986 | 0.0102 | 0.0283 | 0.0153 | 0.35 |

**porcelain**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | wide / input |
|---|---|---|---|---|---|
| marbles-t002 | 0.00013 | 0.00014 | 0.00015 | 0.00015 | 0.87 |
| marbles-t004 | 0.00142 | 0.00143 | 0.00219 | 0.00154 | 0.65 |
| marbles-t006 | 0.00452 | 0.00444 | 0.00760 | 0.00508 | 0.59 |
| marbles-t008 | 0.00542 | 0.00557 | 0.0101 | 0.00688 | 0.54 |
| marbles-t030 | 0.00326 | 0.00337 | 0.00577 | 0.00361 | 0.57 |
| marbles-t060 | 0.00281 | 0.00288 | 0.00514 | 0.00305 | 0.55 |
| marbles-t120 | 0.00305 | 0.00318 | 0.00536 | 0.00346 | 0.57 |

**table**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | wide / input |
|---|---|---|---|---|---|
| marbles-t002 | 0.00004 | 0.00004 | 0.00016 | 0.00006 | 0.26 |
| marbles-t004 | 0.00024 | 0.00025 | 0.00093 | 0.00035 | 0.26 |
| marbles-t006 | 0.00054 | 0.00056 | 0.00199 | 0.00078 | 0.27 |
| marbles-t008 | 0.00067 | 0.00069 | 0.00291 | 0.00104 | 0.23 |
| marbles-t030 | 0.00072 | 0.00073 | 0.00173 | 0.00067 | 0.42 |
| marbles-t060 | 0.00072 | 0.00073 | 0.00194 | 0.00070 | 0.37 |
| marbles-t120 | 0.00068 | 0.00070 | 0.00184 | 0.00066 | 0.37 |

#### Displayed MSE against the 65,536-sample references (the sequence's first frame, after 8 warm-up frames)

**marbles-t008, frame 480**

| region | wide network | small network | input, bilinear | OIDN, bilinear | 4,096-sample target |
|---|---|---|---|---|---|
| whole | 0.00195 | 0.00203 | 0.00687 | 0.00313 | 0.00018 |
| glass+metal | 0.0119 | 0.0125 | 0.0393 | 0.0202 | 0.00093 |
| porcelain | 0.00516 | 0.00528 | 0.00953 | 0.00633 | 0.00053 |
| table | 0.00064 | 0.00066 | 0.00295 | 0.00099 | 0.00008 |

**marbles-t120, frame 7200**

| region | wide network | small network | input, bilinear | OIDN, bilinear | 4,096-sample target |
|---|---|---|---|---|---|
| whole | 0.00163 | 0.00170 | 0.00462 | 0.00221 | 0.00019 |
| glass+metal | 0.00949 | 0.00988 | 0.0280 | 0.0151 | 0.00125 |
| porcelain | 0.00273 | 0.00286 | 0.00488 | 0.00330 | 0.00056 |
| table | 0.00066 | 0.00069 | 0.00187 | 0.00065 | 0.00005 |

#### Flicker: mean absolute displayed change, frame to frame (7 pairs a sequence)

**whole**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | targets |
|---|---|---|---|---|---|
| marbles-t002 | 0.00031 | 0.00026 | 0.00079 | 0.00030 | 0.00042 |
| marbles-t004 | 0.00297 | 0.00296 | 0.00678 | 0.00303 | 0.00421 |
| marbles-t006 | 0.00653 | 0.00640 | 0.0147 | 0.00637 | 0.00733 |
| marbles-t008 | 0.00875 | 0.00864 | 0.0208 | 0.00960 | 0.00738 |
| marbles-t030 | 0.0149 | 0.0149 | 0.0234 | 0.0106 | 0.0123 |
| marbles-t060 | 0.0149 | 0.0149 | 0.0220 | 0.0103 | 0.0115 |
| marbles-t120 | 0.0151 | 0.0151 | 0.0230 | 0.0108 | 0.0119 |

**glass+metal**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | targets |
|---|---|---|---|---|---|
| marbles-t002 | 0.00130 | 0.00087 | 0.00207 | 0.00049 | 0.00188 |
| marbles-t004 | 0.0107 | 0.0105 | 0.0195 | 0.00920 | 0.0143 |
| marbles-t006 | 0.0218 | 0.0208 | 0.0415 | 0.0230 | 0.0240 |
| marbles-t008 | 0.0292 | 0.0283 | 0.0619 | 0.0392 | 0.0232 |
| marbles-t030 | 0.0297 | 0.0294 | 0.0576 | 0.0252 | 0.0299 |
| marbles-t060 | 0.0275 | 0.0271 | 0.0506 | 0.0226 | 0.0278 |
| marbles-t120 | 0.0280 | 0.0277 | 0.0532 | 0.0264 | 0.0285 |

**porcelain**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | targets |
|---|---|---|---|---|---|
| marbles-t002 | 0.00010 | 0.00006 | 0.00030 | 0.00013 | 0.00036 |
| marbles-t004 | 0.00261 | 0.00266 | 0.00776 | 0.00359 | 0.00711 |
| marbles-t006 | 0.00583 | 0.00556 | 0.0223 | 0.00948 | 0.0181 |
| marbles-t008 | 0.00720 | 0.00695 | 0.0269 | 0.0117 | 0.0172 |
| marbles-t030 | 0.00601 | 0.00588 | 0.0249 | 0.00836 | 0.0176 |
| marbles-t060 | 0.00561 | 0.00556 | 0.0232 | 0.00817 | 0.0160 |
| marbles-t120 | 0.00558 | 0.00556 | 0.0231 | 0.00716 | 0.0162 |

**table**

| sequence | wide network | small network | input, bilinear | OIDN, bilinear | targets |
|---|---|---|---|---|---|
| marbles-t002 | 0.00020 | 0.00019 | 0.00066 | 0.00029 | 0.00025 |
| marbles-t004 | 0.00208 | 0.00209 | 0.00525 | 0.00228 | 0.00290 |
| marbles-t006 | 0.00478 | 0.00475 | 0.0112 | 0.00429 | 0.00489 |
| marbles-t008 | 0.00643 | 0.00641 | 0.0157 | 0.00604 | 0.00508 |
| marbles-t030 | 0.0136 | 0.0136 | 0.0193 | 0.00899 | 0.0100 |
| marbles-t060 | 0.0138 | 0.0139 | 0.0186 | 0.00895 | 0.00937 |
| marbles-t120 | 0.0140 | 0.0141 | 0.0194 | 0.00920 | 0.00974 |


### What the numbers say

- **Against the targets the network takes out about two thirds of the
  input's displayed error** over the whole frame (wide 0.29 to 0.39 of the
  input's; small 2 to 5% above wide), and beats Open Image Denoise on the
  whole frame in every sequence (by 18 to 38%), on glass and metal by 24 to
  41%, and on porcelain by 8 to 21%. On the table OIDN is slightly ahead
  once every firefly is awake (t = 30 to 120: 0.00066 to 0.00070 against
  the wide network's 0.00068 to 0.00072), and behind everywhere else.
- **Against the references:** at t = 8 the wide network's whole-frame
  error is 0.00195 (input 0.00687, OIDN 0.00313), at t = 120 0.00163 (input
  0.00462, OIDN 0.00221).
- **The targets are not what limits it yet.** The 4,096-sample targets'
  own error against the references is 8 to 21% of the network's, region by
  region (porcelain at t = 120 the highest): glass and metal 0.00093 (t = 8) and 0.00125 (t = 120) against
  the network's 0.0119 and 0.0095. The design asked for the glass's target
  error beside the network's: it is a tenth of it. The network's error is
  its own, not inherited from the targets' speckle; heavier targets (the
  user's planned next investment) would lower a floor the network is still
  far above.
- **Glass and metal is still where the error is:** 0.0095 to 0.0125 there
  against 0.0007 on the table; the network takes 64 to 70% of the input's
  error off it at t >= 8, its best ratio, but from the highest start.
- **The opening** (t = 2 to 6; t = 2 is earlier than any training
  sequence, the earliest of which starts at 2.97 s): the same ratios as later, 0.36 to 0.38 of the input's
  whole-frame error; at t = 2 both errors are tiny (the frame is nearly
  black).
- **Flicker:** on glass and metal the network's frame-to-frame change
  matches the targets' (t = 120: 0.0280 against 0.0285); on porcelain it is
  a third of the targets' (whose own noise flickers); in the opening it is
  at or below the targets'. **On the table, once every firefly is awake,
  it is 36 to 47% above the targets' (0.0136 to 0.0141 against 0.0094 to
  0.0100)
  and above OIDN's (0.0090 to 0.0092)**: low-frequency boiling of the
  table's light from frame to frame, which the per-frame error does not
  show. The same excess showed on the validation split before the test
  (the dry run: table 0.0130 to 0.0159 against the targets' 0.0081 to
  0.0117). The temporal term, under 3% of the loss, is the likely lever:
  Chaitanya et al.'s weights were set "to approximately equalize the
  scales" of their terms, and here the scales are not equal. Raising it
  trades some per-frame error for stability; a decision for the user, with
  its measurement (queued).
- **Small against wide:** the small network's displayed error is above the
  wide one's in 24 of the 28 sequence-and-region cells, by 1 to 9% (the
  whole frame by 2 to 7% from t = 4 on; glass and metal at t = 120 0.0102
  against 0.0099; against the reference at t = 8 0.0125 against 0.0119),
  and below or equal in the other four (at t = 2, where both are near zero,
  and porcelain at t = 6); its flicker within 5% of the wide one's from
  t = 4 on. For 54% of the
  parameters and 56% of the U-Net's multiply-adds. Which one Phase 4 runs
  is the user's call with Phase 4's Metal cost in hand: both are exported.

## Images

Side by side (input upscaled bilinearly | denoised | target), displayed,
5760 x 1080, for each network: the opening at t = 4 (frame 240), t = 30
(frame 1800) and t = 120 (frame 7200), and an MP4 of marbles-t120's 8
frames at 8 frames a second (3840 x 720, each panel scaled to 1280 x
720), not committed.

## The export

`export.py` into `../serenity-data/denoiser-v1-model/{small,wide}/`:
`weights.bin` (float32, PyTorch's tensor order), `layout.json` (each
tensor's name, shape and offset; the structure layer by layer; the 28
input channels and how each is made; the 124 scores and their order; the
constants: albedo floor, PU's coefficients, the masked score; the
provenance: checkpoint SHA-256, epoch, validation loss, commit, data set
manifest) and `SHA256SUMS`. Each export read its weights back by the layout
alone and computed what the checkpoint computes, bit for bit.

| | weights.bin | SHA-256 (weights.bin) |
|---|---|---|
| small | 1,176,720 bytes | adb31169482d5e4404f67fef54ef8fc3b460d57d9dbbe933c69c06071334d6cb |
| wide | 2,170,992 bytes | 6f31158ceb8ad41527b11472e20079267287be9e6f34cb44b5c01860987dd4af |

## Tests

`test_training.py`, 13 tests on synthetic data: the PFM row order; the
warp's convention, "old = new + MV", and its edges; that every flip and the
transpose keep the motion true (the warp still lands after augmenting both
frames) and mirror the normals as stated; one color permutation for input
and target, radiance and albedo; the center tap alone is nearest-neighbor
upsampling times the albedo factor; taps outside the image get no weight;
each of the four output pixels over an input pixel can choose its own tap;
history blends only where valid; PU monotone and normalized; MS-SSIM of an
image with itself is 1; Chaitanya et al.'s frame weights. Each convention
was broken once on a copy to see a test fail: the motion's sign (2 tests
fail), dx kept under a horizontal flip (1), the transpose without its
negation (1), the target's albedo not permuted (1), taps unmasked (2), the
output multiplied by the bare albedo (1), alpha ignoring validity (2),
frame weights of sigma 1 (1).

## Anything odd

- **ReSTIR PT's radiance is not bit-reproducible on the GPU.** Rendering a
  test sequence again with the data set's own binary and arguments gives
  features identical to the byte and radiance that differs in 1 to 9
  pixels of some frames, by at most 2.4e-7: float rounding, from some
  order-dependent sum, not a different frame. Two reruns differ from each
  other too. The data set note's reproduction of one sequence byte for byte
  was that sequence's luck. `render_warmup.py` checks the features exactly
  and the radiance within 16 pixels and 1e-5 (queued).
- **Thomas et al. 2022 could not be read:** the paper's PDF (121 MB) was
  past what could be fetched, and the Eurographics copy refused the
  request. Their loss weights and their handling of the sub-pixel offset
  are not checked against this (queued).
- **The first small run's divergence** (above) was at a rate the 15-epoch
  trial had trained through; 15 epochs do not show what 300 at that rate
  do.
- **No serenity window or renderer was open during the runs and the
  timing;** the two long runs shared the GPU with each other and, for a
  stretch around epoch 170, with something else that made epochs 3 to 7
  times longer (swap was full); the curves are by epoch, not by time, so
  that does not move them.

## Queued

In QUEUE.md:

- The table's flicker: the temporal weight at 0.125 x 2, x 4, x 8,
  validation flicker against validation error, before the next run (a
  decision for the user).
- A longer cycle (600 epochs) to see whether validation keeps falling.
- Phase 4: the network's normal in view space, by the camera's axes (or
  the features pass writes view-space normals).
- Phase 4: small or wide, by the Metal cost; both exported.
- ReSTIR PT's radiance not bit-reproducible: find the order-dependent sum.
- Thomas et al. 2022's loss weights and sub-pixel handling, from the paper.
