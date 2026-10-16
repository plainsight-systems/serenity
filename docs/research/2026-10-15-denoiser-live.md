# The denoiser, live: Phase 4

2026-10-15. Phase 4 of Milestone 3 (2026-10-14-denoiser-training-design.md,
"Phases"): the trained network running in the renderer every frame, so the
window shows it. Apple M3 Max, the toolchain in `cmake/toolchain.json`,
release build.

## What was built

- **A `denoise` pass** (`src/metal/passes/denoise/`; contract 18,
  `src/core/contracts/denoise.h`): `training/model.py`'s `Denoiser.frame`,
  with `training/data.py`'s input channels and `training/evaluate.py`'s
  zero padding to a multiple of 16, as it was trained, on Metal compute
  kernels of our own inside the frame's one compute encoder. 23
  dispatches: the depth median (exact, as `np.median`: three levels of
  radix selection over the float's bits), the 28 input channels with last
  frame's output moved by the motion vectors, the U-Net's eleven 3 × 3
  convolutions and four max pools, and one output kernel for the 1 × 1
  head, the three softmax kernels over the demodulated light, the spacing
  mix, the history blend and the remodulation.
- **Two resolutions in one frame.** A graph with the pass renders every
  pass before it at the output halved each way, rounded up
  (`frame::light_size`), and the denoise and presenting passes at the
  output's size. The renderer keeps a second ring of frame constants and
  camera for the output size, so a graph without the pass binds and
  computes what it did. In the window the light is traced at a quarter of
  the window's pixels and the network brings it to every pixel; `--scale`
  is no longer needed for that (it still applies on top).
- **The full-size albedo** the output is remodulated by: the features
  kernel dispatched a second time, at the output's size, as the training
  data's full-size albedo was rendered (`graphs/features.toml` at 1920 ×
  1080 beside the 960 × 540 light). The cheapest correct alternative
  considered: an albedo-only variant of that kernel; the rays are the cost
  and it would trace the same ones, so not built (queued, with its
  measured cost below).
- **The weights are read at start-up** from `training/export.py`'s
  directory (`weights.bin`, `layout.json`) by the core
  (`src/core/denoise/network.h`), every constant of the pre- and
  post-processing checked against contract 18 and every tensor's shape
  against the widths; a ~200-line strict JSON reader of our own
  (`src/core/denoise/json.h`) rather than a new dependency for one file.
  Both trained networks load from their exports; the graph's `[denoise]
  weights` setting names which.
- **`graphs/restir_pt_denoised.toml`**: ReSTIR PT + MNEE + features at the
  light size, then denoise, then the tone map.
- **History.** The pass's output feeds the next frame, so a graph with it
  reuses history (`frame::reuses_history`) and the denoiser's buffers join
  the core's `ReuseHistory` as ReSTIR's reservoirs do: at a history's first
  frame, after a resize and after a frame out of order, it starts over,
  with no history read (`metal/frame/denoise_images.h`).
- **`training/denoise_check.py`** and **`tests/gpu/denoise_check`**: a
  fixture of the renderer's own images and PyTorch's outputs for them,
  run through the GPU's pass and compared.

No new dependency: the C++ side reads JSON itself, and `denoise_check.py`
uses only what `training/uv.lock` already pins.

## The path chosen, and why

The convolutions could run three ways on this machine. The kernels that
apply the softmax weights and blend the history are ours in every case.

| Path | For | Against | |
|---|---|---|---|
| Metal 4's machine-learning encoder (`MTL4MachineLearningCommandEncoder`) running a Core ML model converted to a Metal package | Apple's route for a network inside a Metal 4 frame; on chips with neural accelerators in the GPU (M5) it would use them | a network is a package built offline (coremltools, a new pinned dependency, and `metal-package-builder`), so the wide and small networks are two conversions, not one setting; the input's dimensions are given to its pipeline's descriptor, so a resize rebuilds the pipeline; the pre- and post-processing are ours anyway, bridged through `MTLTensor`s; on the M3 Max it runs GPU shaders we cannot see, tune, or make match PyTorch other than by trying; and the Vulkan backend to come has no Core ML | not chosen |
| MPSGraph | tuned convolutions; Open Image Denoise's Metal device uses it | Objective-C only, and it encodes into its own (Metal 3) command buffers, not Metal 4's: a second queue and events between it and the frame | not chosen |
| **Our own compute kernels** | one encoder, one argument table, the frame's barriers; weights read at run time; every number ours to match PyTorch with; ports to the Vulkan backend (cooperative matrices) as the rest does | the convolutions must be made fast by us | **chosen** |

Production practice agrees as far as it is public: the shipping real-time
neural upscalers and denoisers (DLSS, XeSS, FSR 4) run their networks as
the vendor's own compute shaders in the engine's command stream, not
through a general inference runtime. The alternative that was not tried
is the machine-learning encoder: on the M3 Max it would have to run the
convolutions faster than the rate below to matter.

## The match against PyTorch

**Precision.** Activations and weights in half floats, every sum in float,
on simdgroup matrices (half × half, float accumulators); the light, its
demodulation (up to 50 times the radiance, past half's range), the
history and everything after the head's sums in float.

**The test** (`tests/gpu/denoise_test.cpp`, in ctest under Metal's API and
shader validation): `tests/data/denoise`, a network of the trained
structure at small widths (8 to 40) with random weights, head included,
and three consecutive frames of a training sequence's renderer images, a
40 × 22 crop where fireflies move (so the padding to 48 × 32, the masked
taps, the depth median, the motion vectors and the history all take
part), against `model.Denoiser.frame` in float32 on the CPU. Largest
relative difference |gpu - torch| / (|torch| + 10^-3), over every frame,
pixel and channel: **2.2 × 10^-4**; the test's tolerance 10^-3. One weight
of the last convolution a tenth off gives 4.0 × 10^-3, and the test that
it fails is in ctest too.

**Real frames** (`denoise_check`, not in ctest: the frames and network are
beside the checkout): the wide network, the 8 frames of validation
sequence gen-0125 (awake: every firefly moving) at 960 × 540 → 1920 ×
1080, recurrently, with the renderer's tone map (`training/tonemap.py`)
on both:

| Frame | Mean relative difference | Pixels differing after tone mapping | Largest difference, 8-bit levels |
|---|---|---|---|
| 0 | 1.6 × 10^-4 | 0.90% (by 1 level, all but 0.001%) | 4 |
| 1 | 1.4 × 10^-4 | 0.82% | 3 |
| 2 | 1.4 × 10^-4 | 0.80% | 6 |
| 3 | 1.3 × 10^-4 | 0.77% | 5 |
| 4 | 1.3 × 10^-4 | 0.75% | 9 |
| 5 | 1.2 × 10^-4 | 0.72% | 4 |
| 6 | 1.2 × 10^-4 | 0.72% | 3 |
| 7 | 1.2 × 10^-4 | 0.70% | 2 |

The small network on the same frames: mean relative difference 0.9 to
1.1 × 10^-4, 0.53 to 0.63% of displayed pixels one level off, none more.

So 99.1 to 99.3% of the wide network's displayed pixels are PyTorch's
exactly; the rest
differ by one level of 255 but for some 20 to 50 pixels in 2 million, by
up to 9 levels at 4 pixels of frame 4, where the PyTorch output differs by
6 to 9% from the GPU's in a pixel of a sharp kernel. The difference does
not grow with the frames: the history carries it, and does not amplify
it. Half precision is a quality trade only there, and it is stated here
rather than taken silently.

Existing graphs are unchanged: test_pattern, preview, path, path_mnee,
restir, restir_ris, restir_spatial, restir_temporal, restir_pt,
restir_pt_features and features, frames 40 and 41 at 480 × 270 on the
marbles, PNG and PFM and features, byte for byte against the build before
this change (`b8e5776`).

## Making it fast

Measured on the pass alone (its inputs arbitrary: its cost does not
depend on them), the wide network at the window's 1728 × 1117 light,
while the small network trained on the same GPU, so only the ratios
count:

| Change | ms | Kept |
|---|---|---|
| First: each simdgroup 32 pixels × up to 6 blocks of 8 outputs, loading straight from device memory | 162 | — |
| Each block of 8 input channels' 6 × 34-pixel halo and weights staged once a threadgroup; at most 4 output blocks a simdgroup (6 spilled: one 48-wide layer took 81 ms) | 57 | yes |
| Max pools and upsamplings with channels fastest (coalesced writes); the head as a matrix product in the output kernel, its scores in threadgroup memory, never in device memory | 42 to 47 | yes |
| The head's weights in half, like the convolutions' (float weights: 9 ms more, the PyTorch match no closer) | 37 | yes |
| No concatenation: a decoder reads the level below upsampled, and its skip, where they are | 30 | yes |
| The output kernel with a scalar head, weights from the constant cache | 56 | no |
| The output kernel's taps from a threadgroup-staged halo | ±0 | no |
| The convolution's halo and result staging aliased in threadgroup memory | ±0 | no |

On the idle GPU, at the end, the pass is 26.4 ms (Frame times, below).

## Frame times

GPU time a frame, median of 179 frames after 21 of warm-up, two frames in
flight, the marbles from t = 120 with time advancing a sixtieth of a second
a frame, the GPU idle before each run (utilization at most 5% for 6 s; no
window, no training), two rounds interleaved (round 1 / round 2). Build
`9033002` plus this note's commit; the base `b8e5776`.

**The denoised graph** (`graphs/restir_pt_denoised.toml`):

| Output (light) | Wide network | Small network |
|---|---|---|
| 3456 × 2234, the window (1728 × 1117) | 79.1 / 77.5 ms | 67.9 / 67.8 ms |
| 1920 × 1080 (960 × 540) | 23.8 / 23.7 ms | 21.1 / 21.1 ms |

**Its parts**, each measured alone (ms, both rounds within 0.1 but where
shown):

| | Window (1728 × 1117 → 3456 × 2234) | 1920 × 1080 (960 × 540 →) |
|---|---|---|
| ReSTIR PT + MNEE + features, at the light size (`graphs/restir_pt_features.toml`, its tone map included) | 41.6 / 41.5 | 13.2 |
| The full-size albedo: the features kernel at the output size (`graphs/features.toml`) | 8.7 | 2.8 |
| **The denoise pass, wide network** | **26.4** | **7.5** |
| **The denoise pass, small network** | **16.7** | **4.8** |

The parts sum to the whole within a millisecond (the tone map at the
output's size is the rest). For comparison, ReSTIR PT alone at the
window's full size is 149 to 150 ms (`graphs/restir_pt.toml`, below): the
denoised graph shows every pixel of the window at about half that, with
the noise taken out.

**Inside the denoise pass**, wide network, window: the depth median 0.5,
the input 0.8, the encoders, pools and bottleneck 6.0 (the first
convolution 2.9), the decoders 13.7 (the last one, at the light size, 7.3),
the last convolution 2.9, the output kernel (head, kernels, blend) 3.0.
The convolutions run at 6.1 × 10^12 multiply-adds a second (the first, the
last decoder and the last convolution alike): the wide U-Net is 70,000
multiply-adds an input pixel, 131 × 10^9 a frame at the window's light,
21.7 ms of the 26.4. As trained it cannot get much cheaper here without a
faster matrix rate; the small network's 37,000 are 16.7 ms.

**Existing graphs, the same time** (ms, base / this change, two rounds):

| Graph, 3456 × 2234 | Base `b8e5776` | This change |
|---|---|---|
| graphs/restir_pt.toml | 149.0 / 150.0 | 149.0 / 150.3 |
| graphs/path.toml | 30.5 / 30.4 | 30.1 / 30.5 |

## Watching it

```sh
make run GRAPH=graphs/restir_pt_denoised.toml SCENE=scenes/marbles.toml
```

The graph names `../../serenity-data/denoiser-v1-model/wide` (relative to
`graphs/`, so from the main checkout). The small network is exported beside
it (`denoiser-v1-model/small`, from `v1-small-lr1e-3/best.pt`, epoch 285):
change the graph's `weights` to `../../serenity-data/denoiser-v1-model/small`.

## Not done

- The full-size albedo traces the features kernel's rays again at the
  output's size: 8.7 ms of the window's 79 (11%). An albedo-only variant
  would save its writes, not its rays.
- The export's `SHA256SUMS` is not checked at load; sizes, shapes, finite
  values and constants are.
- Specular motion vectors (Phase 1's note): the history through glass and
  in mirrors moves with the glass's surface.
