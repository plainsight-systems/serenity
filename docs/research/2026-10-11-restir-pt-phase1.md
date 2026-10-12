# ReSTIR PT Phase 1: initial sampling and paired spatial reuse

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, release builds
(`-fmetal-math-mode=fast`), `scenes/marbles.toml`. Base is b57e76d (the
design, `2026-10-11-restir-pt-layout.md`); Phase 1 is da2e357 and 223e257.*

## What was built

As the design lists it, ReSTIR PT Enhanced at its defaults:

- **Initial sampling** (`restir_pt_initial`): the path loop with a third
  template parameter, what it does with each light found
  (`metal/integrator/path.metal.h`, "What the loop does with the light it
  finds"); the path tree's gather (`metal/passes/restir_pt/restir_pt.metal.h`)
  streams every candidate path into one 64-byte path reservoir
  (contract 15): NEE by RIS at every vertex, 32 / B^2 candidates, the
  M-sample balance heuristic (`metal/integrator/nee.metal.h`); one lobe per
  vertex; at most 9 surfaces; roulette's survival in the source density;
  direct and indirect light in one reservoir.
- **The hybrid shift** (`metal/integrator/shift.metal.h`): replay from the
  destination's own primary hit with the source's numbers at fixed
  dimensions, reconnection to the stored x_k (a primitive and a point in the
  shape's own coordinates, `metal/shapes/shape_point.metal.h`), eq. 4's
  Jacobian, the invertibility check, forced NEE reconnection; the dual
  footprint criteria (c = 0.02, alpha_min 0.2 at x_{k-1},
  `metal/integrator/reconnection.metal.h`).
- **Paired spatial reuse** (`restir_pt_spatial`): three reuse textures (254,
  230, 210; sigma 16) built once on the host (`core/sampling/pairing.h`),
  flipped, mirrored, transposed and offset per frame
  (`metal/reuse/pairing.metal.h`); a shift prepass, the replaying pairs
  compacted (one atomic a SIMD group) into an indirect dispatch, then the
  pairwise MIS merge summed vector-valued for shading
  (`metal/reuse/path_merge.metal.h`). `graphs/restir_pt.toml`.
- Departures, each stated where it is made: the reservoir's second seed
  word unused (a vertex's numbers' dimension is fixed by its index);
  eq. 3's signs read as (+, -, +); the similarity test made symmetric; the
  primary hit traced again by every shift rather than read from a G-buffer.

The path pass keeps its image to the bit (PFM compared against b57e76d's
binary at t = 120) and its time: a first version of the gather was 1.4%
slower with the same bits (the LLVM IR showed a hook's argument built early
and roulette's division hoisted); 223e257 restores the IR.

| Path pass, 3456 x 2234 | t = 120 (ms) | t = 8 (ms) |
|---|---|---|
| b57e76d | 30.39, 30.38, 30.36 | 32.54, 32.58, 32.51 |
| 223e257 | 30.42, 30.34, 30.33 | 32.49, 32.65, 32.46 |

## Tests, each shown able to fail

`tests/gpu/restir_pt_test.cpp` and `tests/restir_pt_core_test.cpp`. Each
was run against a mutation of the code it holds, and failed:

| Test | Mutation that failed it |
|---|---|
| initial sampling alone: floors under one and two fireflies (3%), white and glass furnaces (2%) | NEE's MIS weight halved |
| spatial reuse keeps a lit floor's mean and the glass furnace's | the canonical pairwise weight left at 1 |
| the same inputs give the same image, to the bit, the merge at work | the reuse textures seeded from the clock; the replay's output scaled by its place in the compacted list |
| a path shifted to its own pixel: F kept, Jacobian 1, in each of the shift's five cases (1% for the stored point's rounding) | the Jacobian's p_k^y factor dropped |
| eq. 4 on a floor under a firefly (to 2 x 10^-3; matches to 6 digits) | the forced NEE Jacobian without G(y -> x_k) |
| the criteria: a far diffuse pair taken, a mirror x_{k-1} refused, a pair 5 mm apart refused under a camera 39 cm off and taken under one 3 cm off, scaled scenes alike, a mirror x_k refused, NEE at x_k skipping the inverse test | the inverse test always skipped; a fixed threshold in place of the primary footprint |
| each frame's pairing an involution over a 512 x 384 image, deviation 16 +- 10% | the frame's transpose not undone on the offset |
| a reuse texture an involution, deviation sigma | a link's far end storing a wrong offset |

Roulette's fold has no test of its own (queued); the marbles' accumulated
mean holds it. On a test floor under 100 lights with one NEE candidate,
spatial reuse cuts a frame's MSE 2.6 times (0.00520 to 0.00197), so the
merge works as pairwise MIS should with 3 neighbors. Every region's mean
after 256 ReSTIR PT frames at t = 8 is within 0.5% of the reference (the
porcelain -0.55%, as is the path tracer's, -0.52%: the reference's own).

## Frame time

201 frames, two in flight, median of frames 21 to 199; GPU idle (at most 5%
for 6 s) and no serenity process before each run; graphs alternated, 3
rounds; the figure is the median of the rounds.

| | 3456 x 2234 | 1728 x 1117 (half scale) | 1920 x 1080 |
|---|---|---|---|
| path, t = 120 | 30.31 | 8.15 | 8.04 |
| restir_pt, t = 120 | 113.49 | 28.98 | 30.07 |
| path, t = 8 | 32.08 | 8.57 | 8.65 |
| restir_pt, t = 8 | 112.93 | 29.14 | 30.23 |

ReSTIR PT costs 3.5 to 3.7 times the path tracer at every size. Initial
sampling alone is about 60 ms at full size (2x the path pass: 32 NEE
candidates at the primary hit, each a random read of one of 616 lights'
records); the spatial pass about 53 ms (3 shifts a pixel, each the
destination's primary ray traced and its BSDF resolved again, a
reconnection's shadow ray, more for the pairs that replay). Enhanced reports
15.5 ms at 1920 x 1080 on an RTX 5880 Ada with light tiles and no
retraced primaries; both are queued.

## Equal frame time, one displayed frame each

1920 x 1080 (the references' size), time frozen, from frame 65536: one
ReSTIR PT frame against the mean of k path frames, k = its frame time over
the path's at 1920 x 1080: 4 at t = 120 (30.07 / 8.04), 3 at t = 8 (30.23 /
8.65). Against the existing references, stale at the cat's-eye rims (this
compares, it does not certify). Regions by a ray cast through each pixel's
center: glass and metal marbles (209k pixels), porcelain (81k), the table
(1.77M); the fireflies seen directly beside them. MSE is linear radiance's;
clamped MSE is with every value clipped at 1, about what the display can
show; relMSE divides by the reference squared plus 0.01.

| t = 120, k = 4 | MSE ReSTIR PT | MSE path | clamped ReSTIR PT | clamped path | path / ReSTIR PT, clamped |
|---|---|---|---|---|---|
| whole | 1.943 | 0.529 | 0.00752 | 0.00833 | 1.11 |
| glass and metal | 3.393 | 1.013 | 0.0198 | 0.0231 | 1.17 |
| porcelain | 1.716 | 0.253 | 0.00285 | 0.00879 | 3.08 |
| table | 0.995 | 0.279 | 0.00515 | 0.00651 | 1.26 |

| t = 8, k = 3 | MSE ReSTIR PT | MSE path | clamped ReSTIR PT | clamped path | path / ReSTIR PT, clamped |
|---|---|---|---|---|---|
| whole | 3.601 | 1.271 | 0.0154 | 0.0136 | 0.88 |
| glass and metal | 3.642 | 1.269 | 0.0418 | 0.0387 | 0.93 |
| porcelain | 4.331 | 2.668 | 0.0272 | 0.0341 | 1.25 |
| table | 1.897 | 0.643 | 0.0105 | 0.0095 | 0.91 |

In MSE (and relMSE) the path tracer wins everywhere by about 3 to 4 times:
its k frames average down the rare bright samples that set MSE, and
ReSTIR PT's one frame does not. In what the display shows (clamped), ReSTIR
PT wins clearly on porcelain at t = 120 (3.1x) and a little at t = 8
(1.25x); on glass and mirrors it is a tie (1.17x at t = 120, 0.93x at t =
8); on the table, slightly ahead at t = 120 and behind at t = 8. The
images, side by side (path left, ReSTIR PT right), are in the session's
scratchpad, not the repository: `restir-pt-phase1/t120-side-by-side.png`,
`t8-side-by-side.png`, and 800 x 600 crops of the marbles' middle.

## The gate

The design's gate: at equal frame time, ReSTIR PT must visibly beat the
path tracer in the glass, the mirrors and the porcelain. **It does not.**
The porcelain is visibly smoother (its diffuse base, lit directly, is what
paired reuse of direct light improves); the glass and the mirrors are not
visibly better, and the error measures agree. Stop before Phase 2, as the
design says, and decide.

Why, as far as Phase 1 shows (inferences, from the measurements above):

1. **The light that sets the error in glass and mirrors is the light Phase
   1 cannot reuse.** A firefly seen in a mirror or through glass, a coat's
   sharp highlight, a caustic: each is a chain of delta (or near-delta)
   lobes ending on a light 1.5 mm across. No pair of its vertices passes
   the criteria, so the shift replays it whole from the neighbor's primary
   hit, where it misses the tiny light: the shift fails, and the pixel keeps
   its own sample, as noisy as a path tracer's. The design predicted this
   for cores and caustics ("partly", temporal reuse keeping a found path);
   it holds for the mirror and through-glass views too.
2. **Spatial reuse alone is a small multiplier.** Three paired neighbors of
   one candidate each cut a test floor's MSE 2.6x; the frame costs 3.7x.
   ReSTIR PT's gains in production come with temporal reuse (a confidence
   cap of 20), which Phase 1 does not have.
3. **The frame is 3.7x the path tracer's**: no light tiles, and every shift
   traces its primary ray again (Enhanced has the first and a G-buffer for
   the second). Both are queued; halving the cost would move the clamped
   comparisons by about 2x, not the MSE ones, and not change point 1.

What would change the verdict: temporal reuse (Phase 2), which keeps a
found glass or mirror path for up to 20 frames while the camera is still
(the marbles' camera is fixed), with the duplication map against its
correlation; and manifold NEE for the caustics and cores. Neither is tuned
here: every setting is Enhanced's default.

## Queued

In `docs/process/QUEUE.md`: a G-buffer for the shift's primary hits; light
tiles for NEE's candidates; a test of roulette's fold; the indirect grid's
barrier stage; the reconnection segment's medium; eq. 3's signs against the
PDF; path_merge's float parameters (I.24); the merge's kept path for Phase
2.
