# The production sweep: depth of field after the denoiser, camera jitter, light tiles by power

2026-10-18, Milestone 4, by the user's choice. A sweep of the frame
against shipped engines found three production steps it lacked; each is
built here as the engine it follows builds it, one commit each on
`m4-production` (86faa11, 24afc05, 8a771e4, from 78f1c06). M3 Max, the
toolchain of `cmake/toolchain.json`, release build.

## 1. Depth of field after the denoiser

**What it follows.** Production renders a pinhole image, denoises and
upscales it, and applies depth of field after, from depth: Unreal's
post-process chain (temporal upscalers "between Depth of Field and Motion
Blur"), and DLSS's and NRD's inputs, which carry no lens effects. The blur
is Jimenez's ("Next Generation Post Processing in Call of Duty: Advanced
Warfare", SIGGRAPH 2014): a presort to half resolution, each tile's
largest near-field radius and its dilation, scatter as gather over his
49-tap disc with the near and far fields kept apart, a 3 x 3 median, and
a composite at full resolution. Two simplifications, both stated in
`src/metal/passes/depth_of_field/depth_of_field.h`: the fields split by
the side of the plane of focus (the circle of confusion's sign) rather
than by depth against the tile's nearest, which needs no depth scale tuned
to a scene; and the presort a 2 x 2 mean, not his 9-tap bilateral
prefilter, the input being denoised already. The largest radius is
Unreal's default, 2.5% of the width (`r.DOF.Kernel.MaxForegroundRadius`
and `MaxBackgroundRadius`); the marbles' lens reaches 11 to 13 pixels at
1080 lines, so the clamp bounds the work and shapes nothing.

**The circle of confusion is the lens's.** `circle_of_confusion_scale`
(`src/core/camera/thin_lens.h`) derives the thin lens's blur, r H / (2 F
tan(fov / 2)) |z - F| / z pixels, and `tests/gpu/camera_test.cpp` holds it
to the lens rays the shaders trace, to 0.1%, at four depths.

**Where the lens stays** (`docs/architecture/logical-overview.md`,
principle 12). A denoised graph traces its light, its features and its
full-size albedo through the camera with its lens taken off
(`frame::traced_inputs`), and must have the depth of field directly after
its denoise pass (the schedule refuses it otherwise, so a lens is never
dropped). Every other graph traces the lens in its rays: the path tracer,
the reference, as Unreal's path tracer renders depth of field in its rays
while its real-time renderer does it after them. The training targets
must be a pinhole's, as the network's output now is, and they are made
by the same path tracer: so the data set's scenes are written without
their lens (`scripts/make_denoiser_dataset.py`, `pinhole_scene`), inputs
and targets alike. `graphs/path.toml` keeps the lens; its image, and
`graphs/features.toml`'s, are byte-identical to 78f1c06.

**The image.** The near marbles' silhouettes, ragged and speckled before
(every lens ray through a defocused edge's pixel struck marble or table at
random, and the network kept the speckle), are smooth discs blurred over
the table behind them; the far fireflies' discs are the same size. The
network's look is provisional: it was trained on lens-blurred input.

## 2. Camera jitter, given to the network

**What it follows.** Temporal upscalers (DLSS, TSR, FSR) take one camera
jitter a frame from Halton(2, 3), with a phase count of about 8 times the
square of the upscale ratio (DLSS's programming guide): 32 here at 2 x 2.
Frame n's offset is Halton(2, 3) at k = n mod 32 + 1, less 1/2
(`src/core/sampling/jitter.h`, constexpr); its 32 phases put every x in
its own 1/32 of the pixel and the first 27 y in their own 1/27, which a
test holds.

**Which rays take it.** Every ReSTIR PT graph's camera rays and its light
features, in the light's ring of frame constants (contract 7); the light
the network reads is ReSTIR PT's, live in the denoised graph and in the
data set by `graphs/restir_pt_features.toml`, which must take the same.
The path tracer keeps each pixel's own random point (its box filter is
what a reference converges to), and so do the features alone, the data
set's full-size albedo, and the denoised graph's full-size albedo: an
output-resolution image, which production upscalers write unjittered.
Motion vectors stay unjittered (NRD's and DLSS's convention), by
construction: both ends are pinhole projections of the surface's point.
`tests/gpu/features_test.cpp` holds every pixel's ray to the frame's
offset.

**How the network reads it.** As two input channels, constant over the
image (28 and 29: contract 18 now has 30 inputs, in the padding the
32-channel layout already had), as DLSS-style upscalers take the jitter
as an input. The Metal input kernel, the PyTorch model
(`training/model.py`, `network_input`), the data (14 channels; the flips
and the transpose carry the jitter as they carry the motion), the export
and the fixture changed together; the fixture was made again from the same
frames, each with a jitter of its own, and the Metal pass matches PyTorch
on it as before.

**The refusal.** Every network exported before has 28 inputs. The loader
refuses it, by name: "the denoiser's network
.../denoiser-v1-checkerboard-model/wide: inputs.channels is not the 30
this renderer runs with". So the demo graph does not run until the
network is retrained. For the frame times below, the demo's network with
its first convolution padded by zero weights for the two channels
(measurement only, outside the repository; the frame's cost does not
depend on the weights' values).

**The image** of step 2 is shown on the network's input alone (it cannot
be run): the light and the albedo feature at 960 x 540, through a pinhole,
before and after. Before, each pixel's ray took its own point, so an edge
was noisy from pixel to pixel; after, every pixel the frame's offset, so
an edge is a clean, aliased step that moves from frame to frame, which the
network, told where, is to resolve over its 32 phases.

## 3. Light tiles drawn by power

**What it follows.** RTXDI's local-light presampling
(`RTXDI_PresampleLocalLights`): the lights' powers written into a PDF
texture at their indices' Z-curve places (the sample's
`PrepareLights.hlsl`, `PolymorphicLight::GetPower`: 4 pi r^2 x pi x the
radiance's luminance for a sphere), its mip chain built, and each tile
entry drawn by descending it, choosing among four children by their
weights (`RTXDI_SamplePdfMipmap`), its inverse pdf stored. A light's
probability for MIS is its texel over the texels' sum
(`RAB_EvaluateLocalLightSourcePdf`). On a Z-curve the four children are
four consecutive indices, so here the chain is a quadtree over the light
indices in one buffer, as sums (`src/core/contracts/light_tile.h`), built
each frame by one threadgroup in a fixed order of additions, and each
light's probability its leaf over the top
(`src/metal/light_selection/power_light.metal.h`). Every density ReSTIR PT
weighs a light by asks that probability: the NEE candidates', the
light-sampling density MIS weighs a light struck by a bounced ray by, the
shifts' and the reconnections'; the temporal reuse evaluates last frame's
paths with last frame's pyramid. A firefly not yet awake is never drawn.
The path tracer keeps the uniform selection: it is the reference.

**Tests.** A million draws over lights of powers 1 : 4 : 1/4 : 9 : 1 fall
within 5 sigma of those shares, never on a leaf of no light, each with the
probability MIS asks (`tests/gpu/restir_pt_test.cpp`); ReSTIR PT's means
still match the closed form on the lit floor, under two fireflies of
equal power and under two of unequal power (5.76 : 1, so the tiles draw
them unequally), and the glass furnace's; and
a path shifted to its own pixel keeps its F with a Jacobian of 1.

## Frame times

The demo graph, the marbles at t = 120, the frame interval as the window
makes frames (two in flight; `serenity-scratch/checkerboard/
overlap_main.cpp`), 180 frames after 21 of warm-up, the median; three
rounds, interleaved, the median of their medians. AC power; before every
run no other serenity, serenity-headless or overlap_main process and the
GPU at most 5% busy for 6 s. Scripts and raw data:
`serenity-scratch/m4-production/`.

| Tree | 1920 x 1080 | | 3456 x 2234 (the window) | |
|---|---|---|---|---|
| 78f1c06, before | 16.59 ms (60.3 fps) | | 56.01 ms (17.9 fps) | |
| 1: depth of field, pinhole light | 17.10 (58.5) | +0.51 | 57.78 (17.3) | +1.77 |
| 2: camera jitter | 17.09 (58.5) | -0.01 | 57.60 (17.4) | -0.18 |
| 3: light tiles by power | 17.37 (57.6) | +0.28 | 58.30 (17.2) | +0.70 |
| all three | | **+0.78** | | **+2.29** |

Four rounds; nine runs another process overlapped (half again as long or
more; one of them named by the agent beside this one, which measured the
window's pacing at the same time) were run again, each after the same
checks, and the overlapped ones kept apart in the raw data. Steps 2 and 3
are timed with the demo's network padded by zero weights for the jitter's
two channels (above); the network's cost is that of its widths.

Each step's own passes, timed alone (per-dispatch timestamps, no two
frames overlapping, step 3's tree): the depth of field 0.70 ms at
1920 x 1080 (presort 0.15, tiles 0.08, dilation 0.02, gather 0.24, median
0.04, composite 0.19) and 2.47 ms at 3456 x 2234 (0.47, 0.10, 0.04, 0.84,
0.18, 0.85), in the image half, which runs beside the next frame's light
half, so the frame grows by less than it; the power pyramid 0.19 ms at
either size (one threadgroup: 616 lights placed and powered, five levels
of sums each after a barrier) and the tiles' draw 0.02 ms. The jitter
costs nothing: it removes two numbers a pixel draws.

## Images

Under `/private/tmp/claude-501/.../scratchpad/m4-production/` (outside
the repository, for the user's review), each at t = 20 and t = 120:

- `step1-depth-of-field-t*-side-by-side.png`, `...-near-marbles-crop-x2.png`:
  the demo at 1920 x 1080, 78f1c06 against step 1, the near marbles a 2x
  crop. The network is the current one, trained on lens-blurred input:
  its look is provisional until it is retrained.
- `step2-jitter-t*-{light,albedo}-side-by-side.png`, `...-crop-x3.png`:
  the network's input (`graphs/restir_pt_features.toml` at 960 x 540,
  through a pinhole), step 1 against step 2.
- `step3-power-tiles-input-t*-light-*.png`: the network's input, step 2
  against step 3; `step3-power-tiles-demo-t*-*.png`: the demo with the
  zero-padded network (measurement only), the same two trees. The mean is
  the same (tests above); the noise moves toward the light of the bright
  fireflies and away from the dim, newly woken ones, whose own glow on the
  table is noisier frame to frame.

## For the user

- **The network must be retrained**, and until it is the demo graph does
  not run (the refusal above). The data set changes in its inputs and its
  targets, so it is rendered whole, into a new directory (DATASET is
  `denoiser-v2`; `--targets-from` is refused for it):
  `python3 scripts/make_denoiser_dataset.py --out ../serenity-data/denoiser-v2`
  (`--estimate` first for the time; denoiser-v1 took 4.0 h). The test's
  references (t = 8, t = 120) must be rendered again through a pinhole,
  from the data set's own copy of the marbles (`training/README.md`).
- **The data set's inputs are not checkerboarded.** Found on the way, not
  changed: since 01d8aae only a graph with the denoise pass traces its
  light checkerboarded and filled, and the data set's input graph,
  `graphs/restir_pt_features.toml`, has none; denoiser-v1-checkerboard
  (renderer 684f105) was rendered by it, so its inputs are every pixel
  traced, not the checkerboard's half and fill the demo hands the network.
  denoiser-v2 would be the same. Making the input graph trace as the
  denoised graph does (the checkerboard and its fill without the network)
  is a change to the frame graph's rule; whether to make it before
  rendering denoiser-v2 is the user's call.
- **The depth of field's quality and cost.** It gathers at half the
  output's resolution, as Jimenez's does, with his median after; inside
  the most defocused near marbles a little half-resolution blockiness
  shows in the 2x crops. Gathering at the output's resolution would cost
  about four times the gather (some 3 ms more at the window's size).
- **Kept as production does, and so stated**: the path tracer, the
  reference, keeps the lens in its rays; the full-size albedo stays
  unjittered; the motion vectors stay unjittered.
