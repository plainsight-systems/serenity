# The denoiser's features: Phase 1

2026-10-14. Phase 1 of Milestone 3 (2026-10-14-denoiser-training-design.md,
"Phases"): the renderer's outputs the network reads, written by the headless
renderer for training, and the scene generator's seed. Apple M3 Max, the
toolchain in `cmake/toolchain.json`, release build.

## What was built

- **A `features` pass** (`src/metal/passes/features/`; contract 17,
  `src/core/contracts/features.h`): per pixel, at the frame's size, four
  GPU images: albedo, world-space normal, linear depth and motion. They stay
  on the GPU (`FeatureImages`, `src/metal/frame/feature_images.h`), where
  Phase 4's network will read them every frame; files are only for training.
- **Its own pass, not part of ReSTIR PT's initial pass.** A graph that does
  not ask for features compiles, binds and dispatches exactly what it did, so
  it costs nothing there (measured below); any light graph can have it; and
  it renders at a size of its own. It retraces the camera ray, the same ray
  the light passes trace at that pixel and frame (their point in the pixel
  and on the lens), so at 960 × 540 each feature describes the ray its noisy
  sample took. The alternative, writing features from the initial pass,
  would have saved one ray a pixel in the light graph at the cost of a kernel
  variant there and no way to render features at another size.
- **`graphs/features.toml`**, the features pass alone, the one graph that
  shows no image; **`graphs/restir_pt_features.toml`**, ReSTIR PT with the
  features beside it, its light identical to the bit
  (`tests/gpu/features_test.cpp`).
- **`serenity-headless --features`** writes `frame-NNNNNN-albedo.pfm`,
  `-normal`, `-depth` (one value in all three channels) and `-motion`
  ((dx, dy, dz)) beside each written frame. The noisy input is `--format pfm`
  of the light graph at 960 × 540; the full-resolution features are a second
  run of `graphs/features.toml` at 1920 × 1080. Each frame is told the frame
  before it (`frame::FrameInputs::previous`), so a run that starts at frame
  600 has correct motion without replaying.
- **`scripts/make_marbles.py --seed N [--out FILE]`**: with no options, the
  real marbles byte for byte; with a seed, a generated scene with its own
  kinds and counts (28 to 40 marbles), sizes, colors, layout, firefly seeds
  and camera (up to 12° around the aim, ±10% distance, ±2 cm height, aim
  moved up to 2 cm, focused on the aim), every marble kept in frame.

## Glass, mirrors and the sky

Production's rule for auxiliary features at delta surfaces, followed as
written:

- **Open Image Denoise** (README, "Albedos" and "Normals"): the albedo of a
  perfect specular dielectric is "the Fresnel blend of the reflected and
  transmitted albedos", and if noise is an issue "split the path into a
  reflected and a transmitted path at the first hit"; "the reflected albedo
  in itself can be used for mirror-like surfaces"; normals likewise, the
  Fresnel blend, not renormalized; a metal's albedo its reflectivity at
  normal incidence; clear coats ignored; absorption counted.
- **NVIDIA's primary surface replacement** (NRD README, "Interaction with
  primary surface replacements"; "Rendering Perfect Reflections and
  Refractions in Path-Traced Games"): the first non-delta surface replaces
  the primary one; where glass splits, one path is followed, "most likely
  based on a threshold of the Fresnel term"; low-roughness mirrors may be
  treated as perfect.

What the pass does:

| At the primary surface | Albedo | Normal |
|---|---|---|
| Matte, coated (porcelain), rough metal | base color; coat ignored; metal f0 | its shading normal |
| Glass (smooth dielectric) | F × reflected + (1 − F) × transmitted, each half followed to the first non-delta surface, through further glass by the branch Fresnel weighs more, absorption multiplied in | the same Fresnel blend |
| Mirror (conductor, GGX α ≤ 0.01: the marbles' brass, steel, copper) | f0 × the reflected surface's | the reflected surface's |
| A firefly | 1 | its normal |
| Sky (or more than 8 delta hits) | 1 | (0, 0, 0) |

Depth is the primary surface's view-space Z (NRD's `IN_VIEWZ`), glass and
mirrors included, so depth and motion describe one point; 0 for the sky
(no surface, as the G-buffer's distance 0; finite, where NRD would take a
value beyond its range).

The renderer has no delta mirror: its least conductor α is 10^-3. The
α ≤ 0.01 threshold is ours, chosen to cover the scene's "mirrors" and nothing
rougher.

## Motion vectors

NRD's `IN_MV`, "old = new + MV", non-jittered, in its 2.5D form ("`.z =
viewZprev - viewZ`"); DLSS's convention agrees (Programming Guide §3.6.1:
render-resolution pixels, pointing to the previous frame). Per pixel: the
primary surface's point projected through this frame's pinhole camera, and
the same point, placed by its shape's previous transform, through the
previous camera; dx, dy their difference in pixels, dz its view depth then
less now. The sky's from its direction, so only a camera's turn moves it.
With no frame before (the first frame), 0. A still point moves by float
rounding, some 10^-6 pixels, not exactly 0.

Not done (QUEUE.md): what is seen through glass and in mirrors moves
differently from the glass's or mirror's own surface (production's specular
motion vectors).

## Tests

Each shown able to fail by a mutation, then restored:

| Test | Mutation | Result |
|---|---|---|
| A matte floor's albedo, normal, depth (2 m exactly); the sky's | depth × 1.001; albedo channels swapped | fails at 2.002; 0.5996 against 0.2 |
| The floor through a glass ball: 0.04 + 0.96 × albedo, normal 0.96 up, depth the glass's | Fresnel weights swapped | 0.968 against 0.232 |
| The floor in a mirror: the floor's albedo and normal | mirror threshold α 0.001 | albedo 1 (f0) against 0.2 |
| A camera moved 0.1 m sideways: every floor pixel +3.297 px, the sky 0 | current less previous | −3.297 |
| A wandering ball: dx, dy, dz from its motion's own positions at t and t − 1/60 | this frame's transforms as last frame's | off by 4.0 px |
| Headless: names, sizes and refusals (`tests/test_headless.sh`) | `frame-N_albedo.pfm` | wrong names |
| `make_marbles.py`: default byte-identical, a seed reproducible, seeds differ (`tests/test_make_marbles.py`) | default seed 8; unseeded generator | each fails |
| ReSTIR PT's light with and without the features pass, to the bit | — | guard |

## Frame times

GPU time a frame, median of 179 frames after 21 of warm-up, two frames in
flight, the GPU idle before each run (no window; utilization at most 5% for
6 s), the marbles. Build 30f8e5d against this change, runs interleaved:

| Graph | Size, t | 30f8e5d (ms) | This change (ms) |
|---|---|---|---|
| graphs/path.toml | 3456 × 2234, t = 120 | 30.18, 30.06, 30.40 | 30.16, 30.04, 30.04 |
| graphs/restir_pt.toml | 3456 × 2234, t = 120 | 149.29, 148.23, 148.30 | 148.68, 148.32, 148.53 |

Unchanged, and their images the same to the bit as 30f8e5d's: path,
path_mnee, restir and restir_pt, frames 40 and 41 at 480 × 270, PFM and PNG
alike, compared byte for byte.

The pass where asked for it (t = 10, two runs each):

| Graph | Size | ms |
|---|---|---|
| graphs/features.toml | 960 × 540 | 1.34, 1.32 |
| graphs/features.toml | 1920 × 1080 | 2.78, 2.79 |
| graphs/restir_pt.toml | 960 × 540 | 12.46, 12.52 |
| graphs/restir_pt_features.toml | 960 × 540 | 13.68, 13.68 |

So the features add 1.2 ms (+10%) to ReSTIR PT at the network's input size;
a fixed part of it, since 1920 × 1080 costs only 2.1 times 960 × 540. The
saving from writing them in ReSTIR PT's initial pass would be the camera
ray, at most this; queued (QUEUE.md) for Phase 4, which measures the frame
the network runs in.

## Examples

The real marbles at t = 10 (960 × 540 ReSTIR PT + features; 1920 × 1080
features), its motion at t = 20 (at t = 10 the fireflies in frame are still
perched), and generated scene 11 at t = 10: rendered for the hand-off, not
committed. At t = 10 the depth of field shows in the features as it does in
the light: each is the one ray of its pixel, through its point on the lens.
