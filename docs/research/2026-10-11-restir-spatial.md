# ReSTIR DI Phase C: the spatial merge

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, release builds
(`-fmetal-math-mode=fast`), `scenes/marbles.toml`. Base is 4cdaee1
(Phase B); Phase C is 6cf8a32.*

## What was built

- **The spatial merge** (`restir_spatial`, `metal/reuse/spatial.metal.h`),
  RTXDI's defaults: k = 1 neighbor drawn uniformly in a 32-pixel disc,
  refused outside the image or on the pixel itself; depth 0.1, normal 0.5
  and material similarity (`similar.metal.h`, shared with the temporal
  merge); normalized by RTXDI's pi / piSum as the temporal merge is, the
  neighbors' counts as stored. It reads the reservoirs the pass before
  wrote (the working array) and writes this frame's final array, so every
  neighbor read sees the same set. Basic mode traces no ray.
- **Raytraced mode runs with it alone** (`graphs/restir_spatial.toml` with
  `bias_correction = "raytraced"`): its rays are this frame's, so nothing
  is missing. With the temporal merge it stays refused by name, as in
  Phase B (last frame's structure is not kept).
- **Final-visibility reuse across the merge** needed no new code: a
  neighbor's survivor arrives with its test moved by the neighbor's offset
  (`reused_from`), and the shade takes it within 4 frames and 16 pixels,
  else traces and writes the result back (RTXDI's `spatialDistance`).
- **Graphs**: `graphs/restir.toml`, ReSTIR DI whole (initial, temporal,
  spatial, shade, tone map), and `graphs/restir_spatial.toml` (no temporal
  merge). The path graph is unchanged.

## Frame time

3456 x 2234, 201 frames with two in flight, median of frames 21 to 199;
GPU idle (Device Utilization at most 5% for 6 s) and no serenity process
before each run; graphs alternated, 3 rounds. Runs whose p10 to p90 spread
was over 1 ms (another session using the GPU: five of round 3's six runs)
are left out.

| Graph | t = 120 (ms) | t = 8 (ms) |
|---|---|---|
| path | 30.43, 30.41, 30.42 | 32.58, 32.55 |
| restir_temporal | 50.43, 50.47 | 50.46, 50.47 |
| restir (whole) | 53.85, 53.86 | 53.83, 53.80 |

The path graph is Phase B's (30.13 to 30.43, 32.53 to 32.58). The spatial
merge costs 3.4 ms at 3456 x 2234, about 0.85 ms at a traced 1728 x 1117.

## Error at equal time

Phase B's method: 1920 x 1080, time frozen, from frame 65536, two frames
in flight, 1 s, against the existing references (stale at the cat's-eye
rims: this compares, it does not certify). Round 1, with round 2 in
parentheses where it differs by more than 2%.

| Graph | t = 120: frames | accumulated MSE | last frame MSE | t = 8: frames | accumulated MSE | last frame MSE |
|---|---|---|---|---|---|---|
| path | 120 | 0.0166 | 2.05 | 114 | 0.0333 | 3.79 |
| restir_temporal | 69 (72) | 0.0305 | 2.88 (2.12) | 72 | 0.0574 | 4.25 (3.87) |
| restir | 65 (68) | 0.0462 (0.0435) | 1.98 | 67 | 0.0641 | 3.95 |

The whole graph's per-frame image is the least noisy at t = 120 and not
at t = 8; at equal time the MIS path tracer still wins, for the reasons
Phase B's note gives (the shared continuation sets a frame's error, and
ReSTIR's passes cost 20 to 23 ms a frame at full resolution).

Image means over 4 s against the reference's: RIS +0.00%, spatial alone
-0.16% (basic) and +0.00% (raytraced), temporal alone +0.02%; the whole
graph -1.45% at t = 120 and -0.33% at t = 8 (-1.68% and -0.50% with
final-visibility reuse off, which brightens). Each merge alone is within
basic mode's bias; together they darken more, the stated direction of
basic mode's bias (`unbiased.metal.h`). Raytraced mode, which would
separate that bias from anything else, is refused for the whole graph
until last frame's structure is kept: queued.

## Tests

`tests/gpu/restir_test.cpp`, each seen to fail under a mutation, then
restored: a neighbor failing the depth or normal test contributes nothing
(the geometric test skipped: fails); M is the own count plus each accepted
neighbor's, within (k + 1)(H + 1) M, a neighbor's survivor carrying its
offset (a count streamed one more: fails); a still floor converges to the
closed form within 3%, whole and spatial alone (piSum without the
neighbors: fails); the whole graph is the same to the bit for the same
inputs (the spatial stream keyed by a buffer's address: fails). Raytraced
spatial under a box over the floor: RIS's lower half within 1.5% (it gave
1.0000; basic 0.9998).

## Queued

In `docs/process/QUEUE.md`, one line each: the whole graph's -1.45% mean
at t = 120 in basic mode, to check against raytraced once last frame's
structure is kept; the spatial pass's 3.4 ms against its counted traffic
(GPU counters).
