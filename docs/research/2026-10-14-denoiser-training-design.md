# The denoiser: training design

2026-10-14. Milestone 3 adds the pass that production ReSTIR always runs
after it: a denoiser, here one that also upscales. This note designs how
it is trained. Running the trained network in a Metal pass is the second
half of the milestone and gets its own note.

Decided with the user, 2026-10-14:

- **A kernel-predicting network.** A small network looks at each pixel's
  neighborhood and outputs a score per neighbor; the scores are
  softmax-normalized and used to average the noisy colors. The learned part
  is how to compute the scores; the scores themselves are computed fresh
  from every frame, as attention computes its weights from the tokens in
  front of it. Real-time denoisers built this way: Hasselgren et al. 2020
  (NVIDIA), Işık et al. 2021, Thomas et al. 2022 (Intel).
- **Upscaling from the start.** The renderer runs at half resolution each
  way; the network outputs full resolution, as Thomas et al. 2022 does.
  This is what brings ReSTIR PT's 150 ms frame toward real time.
- **PyTorch** for training, for the reference code and tooling around it
  (Intel Open Image Denoise's training toolkit is PyTorch). MLX stays an
  option: it would let one Metal kernel serve both training and the
  renderer.
- **The real marbles scene is never trained on.** It is the test set.

## What the network sees, from the scene

The input is what the renderer will produce at runtime: ReSTIR PT with
MNEE, one path a pixel, at half resolution. Its noise is not the
independent per-pixel noise most denoisers are trained on. Spatial and
temporal reuse correlate neighboring pixels and consecutive frames, so the
network is trained on ReSTIR's own output, not on a path tracer's.

In the marbles, the hard regions for a denoiser are known from Milestone 2:

- fireflies seen directly through glass or in metal, bright points smaller
  than a pixel (58 to 78% of the glass and metal error);
- caustics and the cat's-eye cores, lit through glass;
- the dark opening, where most of the frame is black and a few fireflies
  wake.

Each is in the training data on purpose (below), and each is a region of
the test report.

**Per pixel, at half resolution:**

| Input | Why |
|---|---|
| Noisy radiance, divided by surface color | The network denoises lighting, not texture; color is multiplied back after, so the wood grain and swirls stay sharp |
| Surface color, normal, depth | Edges the filter must not blur across |
| Motion vector | Where this pixel was last frame |
| Last frame's output, moved by the motion vectors | Temporal stability |

**Output, at full resolution:**

- per output pixel, scores for its input neighbors at three widening
  spacings (3 × 3 at spacings 1, 2 and 4), each set softmax-normalized and
  applied to the noisy lighting;
- a weight for blending with last frame's moved output;
- the result multiplied by full-resolution surface color.

The renderer does not yet output motion vectors, or surface color,
normal and depth as images. They are the first thing built (Phases,
below).

## The data

**Training scenes are generated, never the marbles.** `scripts/make_marbles.py`
places the marbles from one fixed seed. It gains a seed option, and the
training set uses 40 scenes from other seeds, each varying:

- the marble layout, kinds and sizes;
- the firefly swarms;
- the camera's position and focus;
- the moment in the animation, from the dark opening to everything awake.

The brass sphere scenes add different geometry and a different scale.

**Each sample is a short sequence.** Flicker can only be learned from
consecutive frames, so each scene contributes 8 consecutive frames, taken
after 30 frames of warm-up so ReSTIR's history is full. For each frame:

- the noisy input: ReSTIR PT + MNEE at 960 × 540, with its features;
- the target: the path tracer at 1920 × 1080, 4,096 samples a pixel.

**Cost.** 40 × 8 = 320 targets at about 8 ms a sample is about 33 s each,
about 3 hours in all. Inputs are a few minutes. One overnight render.

**Splits.**

| Set | What | Used for |
|---|---|---|
| Training | 36 generated scenes | Learning |
| Validation | 4 generated scenes | Choosing when to stop |
| Test | The real marbles at t = 8, 30, 60 and 120, and the opening at t = 2 to 10 | The reported result, looked at once |

## Not overtraining

- **Watch the gap.** Training and validation loss are logged every epoch.
  Training stops at the best validation loss, not at the last epoch.
- **Augment as Open Image Denoise does:** random 256 × 256 crops (128 × 128
  at the input's half resolution), flips, and shuffled color channels and
  normal axes, so the network cannot memorize colors or orientations.
- **Keep the network small.** It must run in a few milliseconds a frame;
  small also means less capacity to memorize.
- **Test once.** The marbles are scored after training ends, not used to
  tune it. If the result is bad, the fix goes into the data or the design,
  and the test is rerun once.

## The loss

Open Image Denoise's default, on brightness passed through its perceptual
curve so one bright firefly does not dominate: 0.16 × mean absolute error
+ 0.84 × (1 − multi-scale structural similarity). Plus a temporal term:
the change between consecutive outputs is penalized for differing from the
change between consecutive targets, which is what stops flicker.

## Training setup

- A `training/` folder outside the C++ build, with its own pinned
  lockfile. PyTorch on the M3 Max's GPU (its `mps` backend); OpenImageIO
  for images. A new dependency, declared as one: it affects nothing the
  renderer builds or runs.
- Open Image Denoise's schedule as the starting point: batch 16, a
  one-cycle learning rate with 15% warm-up.
- The trained weights exported as a plain binary file the renderer loads,
  with the network's layout written beside it.

## How it's judged

On the held-out marbles, against 65,536-sample references, at equal frame
time:

- displayed error, as Milestone 2 measured it, over the whole frame and
  each region;
- flicker: frame-to-frame change of the output against the reference's;
- compared with ReSTIR PT alone, with Open Image Denoise (a single-frame
  quality yardstick), and with MetalFX if its denoiser is available.

## Phases

1. **Renderer outputs.** Motion vectors; surface color, normal and depth
   images; the headless renderer writing them beside each frame; the scene
   generator's seed option. Each output tested.
2. **The data set.** The overnight render, its manifest and checksums.
3. **Training.** The `training/` toolkit, the run, the validation curve.
4. **The Metal pass.** Its own note: the network as Metal kernels, its
   cost, the demo graph `graphs/restir_pt_denoised.toml`, and the test
   results.

## Not decided here

The network's exact size and layout are set by the runtime budget, which
Phase 4 measures. The feature extractor starts as a small U-Net, as Open
Image Denoise's and Thomas et al. 2022's do, and is sized down to fit.
