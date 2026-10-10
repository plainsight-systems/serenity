# The path pass's second write of the mean

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864, `-std=metal4.0`. Release builds.*

## Question

The path pass folded each sample into the accumulated image and then wrote
the same mean again, into a radiance image of its own, for the presenting
pass to read (GDSA.6: no data written twice). Letting the passes after it
read the accumulated image instead saves the write, 16 bytes a pixel, and
the image, 123 MB at 3456 x 2234. What does it save a frame?

## Method

The path tracer (`graphs/path.toml`) on `scenes/marbles.toml` at 3456 x
2234: 200 frames committed back to back with frames in flight, each
settled frame's command-buffer GPU start to end from Metal 4's commit
feedback, the median of frames 21 to 199. Two binaries, before (0b0891b's
renderer with this branch's renames) and after (the accumulated image as
the radiance), alternated, each run started only once no other serenity
process was running and the GPU had reported itself idle (`ioreg`'s
Device Utilization at most 5%) for six seconds. Three sessions, ten pairs.

## Results

| Session | Before (ms) | After (ms) | After - before |
|---|---|---|---|
| 1, before first | 28.535, 28.490, 28.503 | 28.426, 28.446, 28.439 | -0.11, -0.04, -0.06 |
| 2, after first | 28.227, 28.202, 28.222 | 28.439, 28.146, 28.155 | +0.21, -0.06, -0.07 |
| 3, before first | 28.254, 28.224, 28.219, 28.214 | 28.130, 28.156, 28.239, 28.238 | -0.12, -0.07, +0.02, +0.02 |

Seven of ten pairs are faster without the second write; the median
difference is 0.06 ms of 28.2 (0.2%), within the spread between sessions
(0.3 ms) and close to that within one (up to 0.2 ms, the first run of
session 2). An earlier spike of the same change, measured in the shader
sweep, gave 0.30 ms (28.75 against 28.46); that did not reproduce.

## Decision

Adopted, for what it removes rather than for its time: an image of the
frame's size (123 MB at the display's size) and its residency for every
graph whose light pass accumulates, one binding, and one write per pixel,
none of which the frame needed. The frame's time is unchanged within what
these runs can tell. Every GPU test, the path tracer's among them,
passes unchanged, as it must: the presenting passes read the same rgb.
