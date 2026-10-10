# What the passes cost a frame

*Figures from 2026-10-08 to 2026-10-10. Apple M3 Max, macOS 26, Xcode 26.2,
Metal compiler 32023.864, `-std=metal4.0`. Release builds. The first three
sections were recorded in the pass headers when each pass was built and
moved here on 2026-10-10, when the headers were held to design (the figures
are as they were recorded; the commits name the code they measured); the
last was measured on 2026-10-10.*

## The preview pass

`graphs/preview.toml` on the first lit scene (`scenes/brass_sphere.toml`: a
brass sphere on a floor, two fireflies), recorded in 5412b4c (2026-10-08):

| Size | GPU time a frame |
|---|---|
| 3456 x 2234 | 78 ms |
| 1728 x 1117 | 20 ms |

Every light is counted at every pixel (the every-light selection), so the
cost grows with the light count: the preview is for scenes of a few lights
(`src/metal/passes/preview/preview.h`).

## The tone-map pass against the display pass

The path tracer at 3456 x 2234, frames in flight, nothing else on the GPU,
the frame's GPU time ending in the tone map (`graphs/path.toml`) against
ending in the display pass, which shows the radiance as it is:

| Scene | Recorded in | Ending in tone_map | Ending in display | Difference |
|---|---|---|---|---|
| `brass_sphere_flight.toml` | 9db8877 (2026-10-09) | 10.87 ms | 10.27 ms | 0.6 ms |
| `brass_sphere_flight.toml` | df9d622 (2026-10-09) | 12.24 ms | 11.62 ms | 0.6 ms |
| `marbles.toml` (its first look) | df9d622 (2026-10-09) | 11.07 ms | 10.45 ms | 0.6 ms |

Step 1's clamp done on four exact reads for each bilinear one of the first
level, instead of on the filtered reads the pass keeps: the tone map then
cost 1.6 ms over the display pass, against 0.6 ms (9db8877's message; its
header said "1.6 ms more"). Not kept.

## The display pass on its own

`src/metal/frame/frame_images.h` said the display pass was kept apart from
the light pass "at a measured 0.6 ms" (d827b2d). No measurement of the
display pass alone was recorded: the 0.6 ms figures above are the tone map's
cost over the display pass, and the header's own derivation, 247 MB of
traffic at the M3 Max's 400 GB/s, also gives about 0.6 ms. The header now
says it is derived, not measured.

## The barrier between frames

The renderer records one barrier before the first pass of a frame that
touches what frames in flight share (the images between passes and the
accumulated image), waiting for the queue's earlier dispatches
(`src/metal/frame/renderer.h`). `frame_images.h` said what it gives up, the
GPU starting a frame's first dispatch before the last frame's final one
ends, "is measured at implementation"; it had not been. Measured 2026-10-10:

The path tracer (`graphs/path.toml`) on `scenes/marbles.toml` (its second
look, c0bc7d1) at 3456 x 2234, 200 frames recorded and committed with two
in flight, as the window records them; GPU time per frame from commit
feedback, the median of frames 21 to 199; wall time per frame over all 200.
A scratch harness, not committed, linked against the renderer's release
library, with and without the barrier (the variant without it is incorrect,
racing on the shared images, and was built only to time). Run in four
rounds, alternating, each run only on an idle GPU (utilization 0%, no other
GPU program, watched for the whole run):

| | GPU time a frame (median) | Wall time a frame |
|---|---|---|
| with the barrier | 27.65 - 27.70 ms | 27.79 - 27.85 ms |
| without it | 66.6 - 78.9 ms | 44.7 - 48.3 ms |

The overlap the barrier forbids is not a gain: two frames' path kernels
running at once take 1.6 to 1.7 times as long a frame as one after the
other. The kernel is limited by occupancy and the ray tracing unit
(`2026-10-10-path-kernel-counters.md`), and a second frame's threads only
compete for them. The barrier stays, and a copy of the images per frame in
flight would buy nothing.

## The submission's wait for feedback

`Submission::settle()`, which `begin()` calls each frame for the submission
two back, waits for its completion event and then polls for its commit
feedback, which Metal delivers on a queue of its own, sleeping 20 us between
looks (`src/metal/device/submission.h`). Counted 2026-10-10 in a scratch
build of the same harness (counters added to `settle()`, not committed), the
path tracer on the marbles as above: all 200 settles slept, 297 sleeps in
all, about one and a half a frame. The run's frames took 27.23 ms of GPU
time and 27.61 ms of wall time a frame, as without the counters: the frame
is the GPU's, and the CPU's few tens of microseconds of polling fall while
the GPU runs the frame in flight. A blocking wait on a second event, which
the feedback handler would signal, would wake the CPU once instead of one
or two times; it would not change the frame. Not adopted.
