# ReSTIR DI Phase B: the BSDF candidate and the temporal merge

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, release builds
(`-fmetal-math-mode=fast`), `scenes/marbles.toml`. Base is 6a01f9f
(Phase A, ray offset); Phase B is dd97261 (BSDF candidate) and 3427e06
(temporal merge).*

## What was built

- **RTXDI's BSDF candidate** (`bsdf_candidates`, default 1; 0 is light
  candidates alone). After the M = 8 light candidates, a direction from
  the first surface's BSDF, one ray as far as RTXDI's 1e-4 BRDF cutoff
  reaches, and the light it strikes stored as a light point. Every
  candidate is weighed by the balance heuristic over the two strategies,
  p = (8 p_l + 1 p_b) / 9 in solid angle (`target.metal.h`,
  `candidate_density`), and the reservoir finishes over all 9.
- **The light strategy's density asked of the point, not the direction**
  (`light_point_pdf`). A BSDF ray strikes a firefly's rim as often as
  anywhere; there the direction's cone test rounded out, the point was
  weighed as if no light candidate could draw it, and it was counted
  twice: the BSDF floor test read 14% bright before this.
- **The temporal merge** (`graphs/restir_temporal.toml`), RTXDI's defaults
  in basic mode: last frame's final reservoir where the surface was
  (its point placed by last frame's transforms, through last frame's
  camera), permuted within its 4 x 4 block, else the 8 pixels around;
  depth 0.1, normal 0.5 and material similarity; history cap 20 x M;
  RTXDI's pi / piSum with last frame's surface as last frame lit it; the
  shade's traced test written back. The core's reuse history (window and
  headless replay) and the renderer's ping-pong halves, last frame's
  transforms, glows and camera.
- **Raytraced mode with a temporal merge is refused by name**: it needs
  last frame's acceleration structure, not kept. `restir_spatial` stays
  refused (Phase C).

## The ring's start, found by the static-scene test

The first version visited the 8 pixels around the reprojected one in a
fixed order. On a floor seen at a grazing angle (`restir_test.cpp`), a
pixel's permuted first try often fails the depth test, and every such
pixel then took the same-side neighbor every frame: histories drifted one
way across the image. The lower half's mean against the RIS graph's:

| Ring | Lower half | Worst band of 6 rows |
|---|---|---|
| fixed order (from up-left) | +5.6% | +10.4% |
| from a start the frame draws, the same for every pixel (kept) | -0.1% | under 1% |
| permutation off (for reference) | under 0.2% | under 0.2% |

RTXDI draws its further tries at random; a frame-wide start costs one
number a frame. Each merge is unbiased by its weights' derivation either
way, so why a fixed order moved the mean is not understood (queued).

With an occluder (a box over the same grazing floor), basic mode's known
bias remains: the lower half 3% darker than RIS, rows at the box's
shadow edge -7% to +10% (final-visibility reuse brightens, the unshadowed
weights darken). On the marbles the temporal graph's mean over 8 s of
frames is the reference's within 0.03% at both times (below).

## Frame time

3456 x 2234, 201 frames with two in flight, median of frames 21 to 199;
GPU idle (Device Utilization at most 5% for 6 s) and no serenity process
before each run; graphs alternated, 3 rounds. Runs whose p10 to p90 spread
was over 1 ms (another session using the GPU) are left out.

| Graph | t = 120 (ms) | t = 8 (ms) |
|---|---|---|
| path, base | 30.13, 30.42 | 32.26, 32.55, 32.55 |
| path, Phase B | 30.13, 30.43 | 32.58, 32.53 |
| restir_ris, Phase A (base) | 37.24, 37.22, 37.21 | 37.30, 37.30, 37.29 |
| restir_ris, BSDF candidate | 44.20, 44.24, 44.22 | 44.21, 44.22, 44.20 |
| restir_temporal | 50.00, 50.47, 50.66 | 50.47, 50.59 |

- The path graph is unchanged.
- The BSDF candidate costs 7.0 ms (+19%): one closest-hit ray a pixel and
  a BSDF density for each of the 8 light candidates. In a build that
  skips the densities of zero-target candidates, `bsdf_candidates = 0`
  ran 39.0 ms, so the ray and densities are about 5.7 of the 7.0 ms; that
  build's early return was itself 0.5 ms slower, and is not kept.
- The temporal merge adds 6.3 ms over RIS.

## Error at equal time

1920 x 1080, time frozen, from frame 65536 (past the references' sample
indices), two frames in flight, against `.cache/references/marbles-t120-
1920x1080-64x1024.pfm` and `-t8-` (rendered before the ray offset change:
stale at the cat's-eye rims, so this compares, it does not certify).
"Accumulated" is the mean of every frame rendered in the budget. "Last
frame" is the image of one frame alone at the end of the budget, the
accumulated image started over every frame while the reuse history
carried on: for `restir_temporal` that is a temporally accumulating
estimator's per-frame image, which is what a moving scene shows; for the
others it is one sample a pixel. MSE over RGB.

| Graph | t = 120, 1 s: frames | accumulated MSE | last frame MSE | t = 8, 1 s: frames | accumulated MSE | last frame MSE |
|---|---|---|---|---|---|---|
| path | 119 | 0.0168 | 2.05 | 114 | 0.0333 | 3.79 |
| restir_ris | 81 | 0.0248 | 1.96 | 81 | 0.0454 | 3.58 |
| restir_temporal | 73 | 0.0288 | 1.79 | 72 | 0.0574 | 4.25 |

A second round gave the same accumulated MSE to 2%, and last-frame MSE
within 9% (single frames are set by their rarest samples). Over 8 s: path 0.0023 and 0.0049,
RIS 0.0032 and 0.0064, temporal 0.0040 and 0.0091; image means within
0.03% of the reference's for all three.

At equal time the MIS path tracer still wins on the marbles. A frame's
error here is set by the continuation's indirect light, which the three
share and ReSTIR DI does not touch, and ReSTIR's passes cost 14 to 20 ms
a frame at full resolution; the temporal graph's per-frame image is the
least noisy at t = 120, by 9 to 13%, and not at t = 8. Its accumulated mean converges more slowly
than RIS's, as correlated frames do.

## Queued

In `docs/process/QUEUE.md`, one line each: raytraced mode with the temporal
merge (last frame's structure); why a fixed-order ring moved the mean;
basic mode's edge bias at grazing views; the path tracer's MIS asking a
light's density of the direction (the rim rounding found here); the BSDF
candidate's cost; ReSTIR's error at equal time against the path tracer;
the boiling filter, still deferred.
