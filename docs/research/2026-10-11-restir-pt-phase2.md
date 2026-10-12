# ReSTIR PT Phase 2: temporal reuse, duplication maps, and two cost fixes

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, release builds,
`scenes/marbles.toml`. Base is 3ea2a9a (Phase 1,
`2026-10-11-restir-pt-phase1.md`); Phase 2 is f0157b1 and 2cf6285. Built by
the user's direction after Phase 1's gate was not met.*

## What was built

ReSTIR PT Enhanced's remaining defaults (`2026-10-11-restir-pt-layout.md`,
"What is built"), and the two production defaults Phase 1 skipped.
`graphs/restir_pt.toml` is now initial, temporal, spatial, shade,
duplication, tone_map; the two passes are no longer refused.

- **Temporal reuse** (`restir_pt_temporal`,
  `metal/passes/restir_pt/restir_pt.metal.h` steps 5 to 7). The primary
  hit's last-frame point (its shape's point placed by last frame's
  transform) through last frame's camera, searched as ReSTIR DI's Phase B
  searches (the pixel, then the 8 around it from a start the frame draws,
  the first whose last-frame primary hit is similar). Two shifts a pixel:
  last frame's final path shifted here in this frame's scene, and this
  pixel's path shifted there in last frame's (last frame's transforms and
  glows), each by the hybrid shift, the replaying ones compacted into the
  spatial pass's indirect grid. Talbot MIS over the two domains (the 2022
  code's TemporalReuse Talbot branch), confidence capped at
  `confidence_cap` = 20 times the own tree's; the kept path's cached
  Jacobian denominator becomes the shifted path's own.
- **Duplication map** (`restir_pt_duplication`, step 13,
  `metal/reuse/duplication.metal.h`): after each frame, each pixel counts
  the final reservoirs in its 17 x 17 neighborhood holding the same numbers'
  stream (seed pixel and frame), D = count / 288, in 24 x 24 threadgroups
  through threadgroup memory; the next frame's cap at the temporal input's
  pixel is lerp(20, 1, D^0.1) (Enhanced §5).
- **The spatial merge keeps a path** (step 11) by its pairwise weights'
  scalar form, for the next frame; shading stays the vector-valued sum.
- **A G-buffer of primary hits** (contract 5's `PrimaryHit`,
  `core/contracts/primary_hit.h`, 48 bytes): the initial pass packs the
  primary hit it finds anyway (point, geometric normal and wo octahedral,
  the packed BSDF, distance, the primary footprint's threshold, flags and
  interior); every shift starts from the destination's record instead of
  tracing its primary ray and resolving its material again.
- **Light tiles** (`core/contracts/light_tile.h`, Enhanced §6.1, Wyman and
  Panteleev 2021): 128 tiles of 1024 lights drawn by the light selection
  each frame, each a 32-byte packed sphere (center, radius, radiance with
  this frame's glow, light and shape ids); each 8 x 8 block draws one tile;
  NEE's candidates draw from it. With uniform selection the tiles buy
  coherence and one load in place of four dependent ones: measured 4.7%
  off the initial pass (60.1 to 57.3 ms at 3456 x 2234), so kept.

Departures, each stated where made: last frame's domain is traced against
this frame's structure (a firefly that crossed a ray last frame and not
this one is answered by this frame's; queued: keep last frame's structure);
a kept path's suffix after x_k stays as its source frame lit it (as the
2022 code; a moved firefly's light lingers after x_k, bounded by the cap);
the G-buffer's packing (normals and wo to 10^-4 radians, BSDF parameters to
halves), as ReSTIR DI's record and production G-buffers have it.

The path pass is the same to the bit (a 1920 x 1080, 8-sample PFM at
t = 120 compared against 3ea2a9a's binary) and in time (table below).

## Frame time

201 frames, two in flight, median of frames 21 to 199; GPU idle (at most 5%
for 6 s, no serenity process) before each run; graphs alternated, 3 rounds,
the median of the rounds. Runs another session disturbed (p90 over p10 by
more than 5%) were run again. Phase 1 is 3ea2a9a's own binary and graph,
measured in the same session.

| ms | Phase 1 | Phase 2 |
|---|---|---|
| path, 3456 x 2234, t = 120 | 30.42 | 30.43 |
| path, 3456 x 2234, t = 8 | 32.54 | 32.56 |
| restir_pt, 3456 x 2234, t = 120 | 114.04 | 128.82 |
| restir_pt, 3456 x 2234, t = 8 | 114.54 | 132.48 |
| path, 1728 x 1117, t = 120 | 8.12 | 8.12 |
| path, 1728 x 1117, t = 8 | 8.63 | 8.63 |
| restir_pt, 1728 x 1117, t = 120 | 29.48 | 33.17 |
| restir_pt, 1728 x 1117, t = 8 | 29.61 | 33.91 |
| path / restir_pt, 1920 x 1080, t = 120 | 8.04 / 30.07 (Phase 1's note) | 8.10 / 33.92 |
| path / restir_pt, 1920 x 1080, t = 8 | 8.65 / 30.23 | 8.70 / 34.84 |

By pass at 3456 x 2234, t = 120, from graphs that stop after each (the
differences; the same method gave Phase 1's split):

| | Phase 1 | Phase 2 | Enhanced at 1080p (Table 1), times 3.72 for the pixels |
|---|---|---|---|
| initial | about 60 | 58.4 | 6.8 x 3.72 = 25 |
| temporal | none | 35.8 (27.0 with no spatial pass after it) | 3.2 x 3.72 = 12 |
| spatial | about 53 | 30.4 | 3.7 x 3.72 = 14 |
| duplication | none | 4.2 | (0.7 at 1080p, its Fig. 5) |
| whole | 114.0 | 128.8 | 15.5 x 3.72 = 58 |

Phase 1's own graph is 114.0 ms before and 88.8 after the two cost fixes:
the G-buffer took 22 ms off the spatial pass (each of its three shifts a
traced primary ray and a material resolve, now a 48-byte read), the tiles
2.8 ms off the initial pass, and carrying y1's shape across the tree's walk
instead of its last-frame point and transforms 1.8 ms more (2cf6285). The
whole graph is 2.2 times Enhanced's per pixel, on a GPU with roughly a
quarter of an RTX 5880 Ada's FP32 throughput (an inference from the vendors'
figures, not measured here). What is left, by experiments at full size
(each a build with one thing removed, its image wrong, its time read): the
spatial pass's 30 ms is 13.7 ms of reconnection shadow rays, 5.7 ms of
replays and the rest its shifts' arithmetic and the merge; resolving x_k's
material (procedural wood, swirls) costs nothing measurable; the initial
pass is 44 ms with one NEE candidate a vertex (14 over the path pass: the
gather) and 57 with 32. The temporal pass is the spatial pass's work at two
shifts a pixel that nearly every pixel takes. Nothing else is far off; the
rest is queued.

## Equal frame time, one displayed frame each

1920 x 1080, the references' size. ReSTIR PT's frame is the last of 31
consecutive frames a sixtieth of a second apart ending at t, its history
carried from the first (the fireflies moving, as the viewer sees it). The
path tracer's is the mean of k frames frozen at t, k its frame time over the
path's at 1920 x 1080: 4 at t = 120 (33.92 / 8.10 = 4.2) and 4 at t = 8
(34.84 / 8.70 = 4.0). Regions and measures as Phase 1's: masks by a ray
through each pixel's center; MSE of linear radiance; clamped, every value
clipped at 1; and, new here, the displayed image's MSE, both images and a
2048-sample path image tone-mapped by the graph's tone_map (exposure 0,
bloom 0.04) and compared in 8-bit sRGB, which is what the viewer sees.

| t = 120, k = 4 | MSE ReSTIR PT | MSE path | clamped ReSTIR PT | clamped path | displayed ReSTIR PT | displayed path | path / ReSTIR PT, displayed |
|---|---|---|---|---|---|---|---|
| whole | 2.060 | 0.529 | 0.00726 | 0.00833 | 0.00663 | 0.01260 | 1.90 |
| glass and metal | 3.639 | 1.013 | 0.0190 | 0.0231 | 0.0370 | 0.0413 | 1.12 |
| porcelain | 1.037 | 0.253 | 0.00259 | 0.00879 | 0.00591 | 0.01775 | 3.00 |
| table | 1.057 | 0.279 | 0.00496 | 0.00651 | 0.00280 | 0.00902 | 3.22 |

| t = 8, k = 4 | MSE ReSTIR PT | MSE path | clamped ReSTIR PT | clamped path | displayed ReSTIR PT | displayed path | path / ReSTIR PT, displayed |
|---|---|---|---|---|---|---|---|
| whole | 3.557 | 0.977 | 0.0122 | 0.0125 | 0.00890 | 0.01304 | 1.46 |
| glass and metal | 3.680 | 0.962 | 0.0401 | 0.0374 | 0.0472 | 0.0465 | 0.99 |
| porcelain | 2.388 | 1.911 | 0.0111 | 0.0337 | 0.0111 | 0.0373 | 3.37 |
| table | 1.911 | 0.499 | 0.00778 | 0.00864 | 0.00398 | 0.00805 | 2.02 |

Phase 1's frames measured the same displayed way (its PNGs, frozen time,
from frame 65536 with no history): ReSTIR PT 0.00788 whole, 0.0411 glass and
metal, 0.00741 porcelain, 0.00370 table at t = 120 (path / ReSTIR PT 1.60,
1.01, 2.40, 2.44); at t = 8, against k = 3, 0.0122, 0.0509, 0.0280, 0.00657
(1.14, 0.94, 1.37, 1.34).

What the numbers say:

- **What the viewer sees, ReSTIR PT wins** on the table (3.2x at t = 120,
  2.0x at t = 8), the porcelain (3.0x, 3.4x) and the whole frame (1.9x,
  1.5x), at a frame that costs 13 to 16% more than Phase 1's; against
  Phase 1 its own displayed error falls 16% (t = 120) and 27% (t = 8), by
  three fifths on the porcelain at t = 8.
- **Glass and mirrors are a tie** (1.12x, 0.99x): their light is a chain of
  delta lobes to a firefly 1.5 mm across, which temporal reuse keeps only
  while a jittered, replayed path still finds it. Manifold NEE's, as Phase 1
  found and the design says.
- **Linear MSE still favors the path tracer** by about 4x everywhere: it is
  set by rare samples of 10^2 to 10^4 that its k frames average down and
  one ReSTIR frame does not. Clamped at 1, ReSTIR PT is ahead on porcelain
  and table and level elsewhere. Each region's one-frame mean is within
  1.2% of the reference but the porcelain's at t = 120 (+6.2%, one frame's
  bright samples; Phase 1's 256-frame means held every region within 0.5%,
  and the temporal tests below hold still floors' means).
- **The fireflies seen directly** are worse in ReSTIR PT's frame (clamped
  0.19 against 0.016): one camera sample a pixel against the path tracer's
  four, on lights a few pixels across. Not reuse's to fix; anti-aliasing's.

Side by side (path left, ReSTIR PT right), in the session's scratchpad, not
the repository: `restir-pt-phase2/t120-side-by-side.png`,
`t8-side-by-side.png`, and 800 x 600 crops of the marbles' middle,
`t120-crop-side-by-side.png`, `t8-crop-side-by-side.png`. The clip, equal
frame time, 1920 x 1080 a side, the opening (t = 0 to 10 s) and t = 117 to
122 s (15 s; the second segment's history carried from frame 0):
`clips/marbles-path-vs-restir-pt-equal-time.mp4` (CRF 16, 520 MB) and
`-small.mp4` (CRF 24, 157 MB), each half labelled with its ms/frame.

## Tests, each shown able to fail

`tests/gpu/restir_pt_test.cpp` and `tests/restir_pt_core_test.cpp`; every
preset's ctest green, Metal's API and shader validation on. Each new test
was run against a mutation, and failed:

| Test | Mutation that failed it |
|---|---|
| the temporal merge counts last frame's path at most c_cap times the own tree's, c_cap = lerp(20, 1, D^0.1), weighs by Talbot MIS and keeps by its pick (constructed reservoirs, F, W, counts, the kept path's denominator, in double) | last frame's count uncapped; D ignored |
| temporal reuse keeps a still floor's mean under two fireflies, alone and in the whole graph with the duplication map (3%), and the glass furnace's (2%): the tolerances initial sampling and spatial reuse are held to | the own path's Talbot weight left at 1 |
| the duplication map on constructed images: a 5 x 5 block of copies 24/288, a weight-0 copy not counted, copies 8 and 16 apart, another frame's stream not a copy, one stream everywhere 1 inside and the window's share at edges and corners, groups cut by the image | the pixel counted as its own copy |
| a shift from the G-buffer's record gives the retraced shift's success, F and Jacobian within 2% for all but 1 in 200 (1,000+ shifts, 100+ past y2) | the record's wo decoded from its normal |
| the whole graph, temporal reuse and the map at work, the same image to the bit | the temporal replay's output scaled by its place in the list |

Phase 1's self-shift test now goes through the G-buffer, as the passes do;
its tolerance went from 1% to 2%, the record's packing (y1's normal to
10^-4 radians turns every replayed direction after it, which a glossy x_k's
density and a deep replay magnify; one glossy path in 89 was 1.1% off).

## Queued

In `docs/process/QUEUE.md`: the tree's gather's 14 ms; the shifts' shadow
rays; the duplication map's 4.1 ms (a 32-bit key in two arrays was slower);
last frame's structure for last frame's domain; the cached suffix under a
moving firefly; a test of the previous position under a moving primary
surface (the marbles' surfaces are still); dual motion vectors. Done from
Phase 1's queue: the G-buffer, the light tiles, the merge's kept path.

Corpus IDs in the headers: the C++ Core Guidelines' (cpp-guidelines server:
C.1, ES.3, ES.45, F.8, I.1, I.5) and the performance corpus's (its HTTP
client: GDSA.3, GDSA.10, GDSA.13, GDSA.16, GPU.2, GPU.3, GPU.4, GPU.8,
CACHE.4, CACHE.5, MEM.4).
