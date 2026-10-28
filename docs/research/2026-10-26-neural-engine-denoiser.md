# The denoiser's network on the Neural Engine: measured, and a design

For the user's review, 2026-10-26. The question: Milestone 6's finished frame
spends about a third of its GPU time on the denoiser's U-Net (Metal 4's
machine-learning path, change 3); the M3 Max has a 16-core Neural Engine
the frame does not use. Would the network run there instead, beside the
GPU? Nothing is built; the measurements below are standalone.

**Built and kept, 2026-10-27 (the user): section 5.** The demo 63.3 to
84.2 fps at 1920 x 1080 and 17.7 to 25.1 fps at 3456 x 2234, the picture
the GPU's within the Neural Engine's rounding at both.

## 1. Measured

The demo's trained network (`serenity-data/denoiser-v2-model/wide`), its
features (`UNet.features`, what change 3 runs) converted by Core ML Tools
9.0 as `training/export_metal.py` converts it (16-bit, iOS 18's operation
set), timed from Swift through Core ML: input and output IOSurface-backed
half arrays (`MLMultiArray(pixelBuffer:shape:)`, `outputBackings`), so
nothing is copied, as a frame would hand them over. AC power, the GPU idle,
median of 30 after 5 warm-ups.

| Light size (output) | Neural Engine | GPU via Core ML | GPU in the frame (change 3) |
|---|---|---|---|
| 960 x 544 (1920 x 1080) | 6.46 ms | 5.97 ms | 5.73 ms |
| 1728 x 1120 (3456 x 2234) | 23.06 ms | 20.85 ms | 20.42 ms |

- Every operation runs on the Neural Engine (Core ML's compute plan, 38 of
  38 at both sizes; none falls back to the CPU or the GPU).
- Its accuracy is the GPU's: 3.8e-3 of PyTorch's largest feature at
  960 x 544 (the GPU's 3.8e-3; change 3's export check 3.3e-3).
- Sizes. The Neural Engine needs its shapes known: a model with flexible
  height and width (`RangeDim`) runs 33.6 and 118.7 ms and takes no output
  buffer of ours (a copy every frame); with Core ML's hint that sizes
  change rarely (`reshapeFrequency = .infrequent`) 10.7 and 50.2 ms. A model
  of enumerated shapes runs at the fixed model's speed (6.45, 23.0 ms) and
  takes our output buffer. Core ML allows 128 shapes a model.
- A ladder of 121 shapes (sides 256, 336, 448, 544, 672, 832, 960, 1120,
  1376, 1728, 2048, every pair): 6.45 and 23.03 ms on its rungs, 13.8 ms at
  1376 x 832; 2048 x 2048 ran 147 ms, so the largest rungs fall partly off
  the Neural Engine. Loading it took 12.3 s the first time (the Neural
  Engine's compile, which Core ML caches; 0.3 s after); each rung's first
  prediction 0.2 to 0.35 s.

## 2. Production practice: not Apple's guidance for rendering

- Apple's session on neural rendering (WWDC26, "Build real-time neural
  rendering pipelines with Metal", session 359) runs networks on the GPU
  only, at three levels: MetalFX Denoising (Apple's own neural upscaler and
  denoiser, from albedo, depth and motion), custom networks in Metal 4's
  machine-learning encoder "in your command buffer in the same pipeline
  without context switch" (what change 3 built), and small networks inline
  in shaders. It does not mention the Neural Engine or Core ML.
- Per-frame networks on the Neural Engine ship in video upscalers, not in
  renderers found: Anime4K-Metal-CoreML (its heavier networks on the
  Neural Engine through Core ML, about 9 ms a frame on an A15), PiperSR
  (44 fps on an M2 Max, a double-buffered CPU to Neural Engine to Metal
  pipeline).

So the Neural Engine is this project's own design on Apple silicon, not
production's; the project's rule makes that the user's call.

## 3. The design, if chosen

Metal's alone; the CUDA backend is not touched.

**The model.** `training/export_metal.py` becomes the Core ML export: the
features at an enumerated ladder of padded light sizes (at most 128;
rungs chosen so each stays on the Neural Engine, checked by the export with
Core ML's compute plan and refused otherwise, and so the demo's light
sizes, 960 x 544 and 1728 x 1120, are rungs), checked against PyTorch as
now, compiled (`xcrun coremlcompiler`) beside the weights. The Metal
package (`unet.mtlpackage`) and Metal 4's machine-learning path are
removed: one route for the network.

**The size.** The light padded to the smallest rung that holds it (the
network already pads to 16). A size past the largest rung is refused at
prepare() (DenoiseError). Padding further than 16 changes the network's
output near the right and bottom edges (more zeros past the image); the
picture is shown side by side before it is kept.

**The memory.** The network's input (30 planes) and features (32 planes)
in IOSurface-backed half pixel buffers, each plane a block of rows
(Core ML's layout), one each, made at prepare(); Core ML reads and writes
them through `MLMultiArray`s over the same surfaces; Metal sees each as a
linear texture over the same surface (Apple's documented
`newTextureWithDescriptor:iosurface:plane:`), the input kernel writing
one, the head reading the other (the head's loads change from a buffer's
`simdgroup_load` to the texture). No copy on either side.

**The frame.** The image half splits around the network:

1. part 1, the main queue: the light half, as now;
2. part 2a, the overlap queue: the depth median and the network's input,
   then the queue signals the event `network_input` = n + 1;
3. a CPU thread (Metal's shared-event listener) wakes at n + 1, runs the
   prediction on the Neural Engine (`.cpuAndNeuralEngine`), and sets the
   event `network_done` = n + 1;
4. part 2b, the overlap queue, after waiting for `network_done` >= n + 1:
   the head and the output, the depth of field, the tone map, the
   presentation.

The overlap queue's order keeps the surfaces' one copy safe: part 2a of
n + 1 follows part 2b of n, which follows the prediction of n. The main
queue's part 1 of n + 1 waits for none of it: that is the gain, the GPU
tracing frame n + 1 while the Neural Engine runs frame n. Frames in flight
stay two.

**Failure.** A prediction that fails still sets `network_done` (the GPU
never waits forever); the failure is kept and the renderer throws it,
naming Core ML's error, at the next frame's prepare(): visible, not a
frame quietly shown undenoised.

**Tests.** The export's check against PyTorch; the denoise fixture test
through the pass on the Neural Engine (1e-3, as now); the rung for a size,
a pure function tested apart; the failure path (a prediction made to fail
throws at the next prepare()). Metal's 94 outputs: the denoised graphs
change by the Neural Engine's rounding, shown and then rebased.

**Cost a frame, on the CPU:** one wake, one prediction call (blocking its
thread for the network's time), one event set.

**Expected, an estimate not measured:** at 1920 x 1080 the GPU's share
from about 16 to about 10 ms, the Neural Engine's 6.5 inside it: some
90 to 100 fps against 62.6; at 3456 x 2234 from about 57 to about 36 ms,
its 23 inside: some 25 to 27 fps against 17.6. Less where the frame's two
halves already hide part of the network, and where the Neural Engine and
the GPU contend for memory bandwidth; a frame more of latency, the CPU's
round trip.

**Risks.** The first launch's 12 s compile (once a model, cached); a 0.2
to 0.35 s hitch at a window size's first frame (prepare() runs the rung
once); other programs' use of the Neural Engine; Core ML's scheduling,
which no event of ours orders.

## 4. The alternatives

- **MetalFX Denoising**, Apple's production answer (section 2): its
  upscaler and denoiser from albedo, depth and motion, which the features
  pass already writes. It replaces the trained network (Milestone 3's), so
  the picture changes most. Measured 2026-10-27
  (`2026-10-27-metalfx-denoising.md`): faster at every size, but it loses
  the caustics and the mirrors' pinpoints; not kept (the user).
- **A smaller network**: fewer widths or levels, retrained; the picture
  the user's call (`no-silent-quality-tradeoffs`).
- **Keep change 3's route**, Apple's custom-network guidance as built.
  Replaced by the Neural Engine (section 5); its Metal package, tensors
  and export removed.

## 5. Built, measured in the frame, kept (2026-10-27)

Built on the branch `neural-engine` from `apple` at `2ba0735`, as section 3
designs it, after a standalone check that the hand-over copies nothing:
Metal writes the input into an IOSurface-backed pixel buffer, Core ML
reads it and writes the features into ours (`outputBackings`), Metal
reads them, and the features are Core ML's own bit for bit, the
prediction 6.57 and 23.37 ms at the demo's rungs (section 1's 6.46 and
23.06 within a run's spread).

**As built, where it differs from section 3:**

- **The ladder's sides** 256, 320, 448, 544, 640, 832, 960, 1120, 1376,
  1728, 2048: 336 and 672 became 320 and 640, so every side is a
  multiple of 32 halves, a row 64 bytes aligned, and Core Video pads none
  (contract 18's pitch; the renderer refuses a pitch that differs).
- **Metal reads the surfaces through buffers**, not linear textures:
  `newBufferWithBytesNoCopy` over the surface's memory, so the input
  kernel and the head address the planes as before, unchanged, and the
  frame's GPU work is the same work.
- **A failed prediction is the submission's failure**, reported when the
  submission is settled as a GPU error is (`Submission::hand_off()`,
  `HandOffReport`), not at the next frame's prepare(): the last frames'
  failures are reported too.
- **The model's default shape is its first, 256 x 256 or larger:** Core
  ML on macOS 26.6 traps in its first prediction when a model's default
  shape is small (32 x 32, 32 x 48 and 64 x 64 measured, beside a
  96 x 128), and runs every one of them when the default is 256 x 256.
  The test fixture's model has the tests' shapes after a first 256 x 256.

**The frame**, the demo graph on the marbles, `--time 120 --times`, 800
frames at 1920 x 1080 and 300 at 3456 x 2234, each run's mean over its
seconds but the first; the network on the GPU the committed `apple`
build; three runs each, alternated, AC power, the GPU idle before every
run, each configuration's runs within 1% of their mean:

| Output | Network on the GPU (change 3) | Network on the Neural Engine |
|---|---|---|
| 1920 x 1080 | 63.3 fps (15.80 ms) | **84.2 fps (11.87 ms)**, +33% |
| 3456 x 2234 | 17.7 fps (56.52 ms) | **25.1 fps (39.89 ms)**, +42% |

Against section 3's estimate: inside it at 3456 x 2234 (25 to 27 fps),
under it at 1920 x 1080 (90 to 100 fps). A first round, with the system's aerial wallpaper running,
gave 59.9 against 82.4 and 17.6 against 24.0 fps, two configurations
spreading past 2%, and was run again.

**Where 1920 x 1080's difference goes** (2026-10-27, at `39b6e5d`). A
scratch probe, never committed: `Submission::settle()` printing each
part's GPU instants, and the CPU's at `begin()` and the commit, in host
time. The demo at 1920 x 1080 as above, the window's 600 frames, its
rate the unpatched build's (11.91 against 11.93 ms); medians:

| | |
|---|---|
| The frame | 84.1 fps (11.89 ms) |
| The light half, on the main queue | 11.27 ms, a ceiling of 88.7 fps |
| The main queue idle between light halves | 0.54 ms |
| The part before the hand-off; the part after | 0.69 ms; 3.93 ms |
| The hand-off's round trip, Neural Engine included | 7.17 ms (the prediction alone 6.57) |
| The CPU's recording, `begin()` to commit | 0.23 ms |

The light half is the frame: section 3's estimate took the network out
as if nothing else bounded it, and the light half alone is past 90 fps.
The gap is two frames in flight's: `begin(n + 1)` waits for frame n - 1,
which ends about when frame n's light half does (0.15 ms before, the
median), then records and commits, and the main queue waits for it.

**Not taken (the user, 2026-10-27):** three frames in flight (Apple's
Metal Best Practices Guide, "Triple Buffering"), at most the 0.54 ms gap
back, 84 to about 88 fps, for about 300 MB at 3456 x 2234 and slots
apart from `CAMetalDisplayLink`'s latency of 1 or 2; and freeing a slot's
light-half resources when its light half ends, the same gap without a
third frame, an inference not measured. The frame's time is the light
half's. The repeat runs and a 3456 x 2234 control the same hour spread
past 8% from their known rates, cause not found, and are not used.

**The picture**, frame 7029, side by side:
[1920 x 1080](2026-10-26-neural-engine-denoiser/gpu-vs-neural-engine-1920x1080.jpg),
[3456 x 2234](2026-10-26-neural-engine-denoiser/gpu-vs-neural-engine-3456x2234.jpg).
At 1920 x 1080 0.3% of pixels differ, by at most 2 levels of 255; at
3456 x 2234 0.2%, 35 pixels by more than 2, at most 4; every channel's
mean the same. The caustics and the mirrors' pinpoints are the GPU's.

**Between rungs**, a light whose aligned size is not a rung is padded
further, and the network's output changes near the bottom and right
edges (section 3 said so). At 640 x 360 the light, 320 x 180, ran at
320 x 192 on the GPU and runs at the rung 320 x 256: the brass sphere's
denoised frames differ in their bottom quarter (rows 266 to 359), by up
to 46 levels, the means within 0.05 of a level
([side by side, with the difference x 8](2026-10-26-neural-engine-denoiser/padding-between-rungs-640x360.jpg)).
Not at the demo's sizes, whose rungs are their aligned sizes.

**The guards.** Every graph's 94 outputs at 640 x 360 the same to the bit
as the committed build's but the denoised graph's on the brass sphere
scenes (11 files, the padding above); the marbles' denoised frames there,
the dark opening, the same; the same 94 twice. Against the PyTorch
fixture 2.8e-4 relative (tolerance 1e-3). The hand-off tested on both
queues (the worker sees the part before's write, the part after the
worker's, a sleeping worker not overtaken; a worker's failure the
submission's; a second hand-off and one with none open refused); the
network's refusals; recording a frame still allocates nothing of ours.
The window run 15 s on the Neural Engine without an error. Three presets
at 0 warnings, ctest and `make check` green. The guideline corpora's
servers were down this session: the new headers were not checked
against them.

## Sources

- Apple, WWDC26 session 359, "Build real-time neural rendering pipelines
  with Metal": <https://developer.apple.com/videos/play/wwdc2026/359/>
- Anime4K-Metal-CoreML: <https://github.com/mstrji/Anime4K-Metal-CoreML>
- ModelPiper, PiperSR on the Neural Engine:
  <https://modelpiper.com/blog/pipersr-44fps-video-upscale-apple-neural-engine>
- Apple, `MLPredictionOptions.outputBackings`:
  <https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings>
- Apple Developer Forums, MLMultiArray over an IOSurface-backed
  CVPixelBuffer (row strides padded to 32):
  <https://developer.apple.com/forums/thread/712709>
