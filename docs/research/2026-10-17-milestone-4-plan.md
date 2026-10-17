# Milestone 4: making the frame fast

2026-10-17. Four standard optimizations, chosen by the user from the
frame's measured costs: async compute overlap, ray masks, a split scene
structure, and checkerboard rendering of the light. Reservoir reuse as
kernels (tiling, layout and fusion of the merge passes) stays in the queue
as its own later item.

## Where the frame goes

The denoised demo (graphs/restir_pt_denoised.toml, the wide network) in
the 3456 × 2234 window, light traced at 1728 × 1117, marbles at t = 120,
M3 Max on AC power. Per-dispatch GPU timestamps, median of three rounds;
limiters from the GPU's Performance Limiters counters. t = 20 is within 2%;
1920 × 1080 output (23.5 ms) scales the same way. Raw data and scripts:
serenity-scratch/frame-cost-map/ beside the checkout.

| Block | ms | Limited by |
|---|---|---|
| Initial path sampling | 15.6 | ray-tracing unit, 91% active |
| MNEE solve | 4.5 | F32 arithmetic 59%, ray tracing 47% |
| Temporal reuse (shift, replay, merge) | 7.3 | ray tracing 91 to 95% in the shift and replay |
| Spatial reuse (shift, replay, merge) | 9.5 | ray tracing 94% in the shift and replay |
| Shade, duplication map | 1.3 | memory, instruction issue |
| Features at the light's size | 2.7 | ray tracing 84% |
| Features at the output's size (for the full-size albedo) | 8.4 | ray tracing 97% |
| Denoiser | 26.4 | F32 arithmetic 97 to 98% |
| Tone map | 1.0 | |
| **Frame** | **77** | about 13 frames a second |

Two facts shape the plan. The light block is bound by the ray-tracing
unit, about 90% busy, while ALU utilization sits at 12 to 15% and
occupancy at 16 to 22%. The denoiser is the reverse: arithmetic-bound with
the ray-tracing unit idle. Run in sequence, each leaves the other's
hardware idle.

## The four optimizations, in order

1. **Async compute overlap.** Frame N's denoiser runs while frame N + 1
   traces its light, on different hardware, so the frame time moves toward
   the slower of the two instead of their sum. The image is unchanged; the
   cost is one frame of latency. Console engines use this routinely.
2. **Ray masks.** 616 of the scene's 653 objects are fireflies 1.5 mm
   across, and every ray traverses them. Visibility rays (NEE shadow rays,
   the shifts' reconnection checks) skip them through Metal's intersection
   masks, as DXR's instance masks and Vulkan's cull masks are used in
   production. The image changes only by the fireflies' own shadows,
   shown to the user before it is kept.
3. **Split scene structure.** The static marbles and table built once, the
   moving fireflies in their own structure, a top level over both, so only
   what moves is rebuilt and rays that only need occluders never touch the
   swarm. The standard two-level layout.
4. **Checkerboard rendering of the light.** Each frame traces half the
   pixels, alternating, and temporal reuse fills the other half, as console
   checkerboard rendering and RTXDI's checkerboard mode do. It changes the
   image and what the denoiser sees: the network was trained on full light
   frames. Built last, judged with the user on the denoised image; if the
   network needs retraining on checkerboard input, the pipeline from
   Milestone 3 does it (data about 4 h on the Mac; training on the RTX 3090,
   below).

Each step follows the project's design order (docs/process/MEMORY.md):
the architecture documents first (docs/architecture/: the logical overview,
the change axes and the file mapping, updated for what the step adds or
moves), then the design written into the headers it touches, with both C++
guideline corpora (the C++ Core Guidelines and the C++ performance
guidelines) checked rule by rule as each header is written and the
deciding rules cited by ID, then the code.

Each change is measured as it lands (per-dispatch timestamps, frame time
at both sizes), kept only if faster, and checked for an unchanged image
except where the change is meant to alter it (2's firefly shadows, 4),
which the user sees first.

## Training on the RTX 3090

Any retraining runs on the user's RTX 3090 Linux machine over SSH, not on
the Mac: it frees the Mac's GPU and should train faster than the M3 Max's
`mps` backend. The data set is rendered on the Mac (the renderer is
Metal) and copied over. This needs the training toolkit's environment to
resolve for Linux with CUDA (its lockfile pinned for both), a declared
dependency change made when the retrain is decided, not before.

## Step 1: async compute overlap

Built 2026-10-17 (src/metal/device/submission.h, "Two parts";
src/metal/frame/renderer.h, "Two halves"). A denoised frame's submission
has two parts. The **light half** (animate, the ReSTIR PT passes, MNEE,
the features at both sizes) goes on the main queue; the **image half**
(the denoiser, the tone map) on a second MTL4CommandQueue, which waits for
the light half of the same frame by a shared event (Apple's way of
ordering work across Metal 4 queues; D3D12's multi-engine fences and
console async compute have the same shape: a narrow edge per frame, not a
queue drain). Frame N's image half can then run beside frame N + 1's
light half. What the image half reads of the light half's is ringed by
the frame slot, so frame N + 1 never writes what frame N reads: the
features at both sizes one set per slot, and the light's radiance copied
into a per-slot image at the end of the light half, since the accumulated
image carries the light's history (332 MB more at the window's size,
made once). Each half keeps its own history in order on its own queue. A
graph without the denoiser submits exactly what it did. The window's
frame time now counts a frame from the end of the one before when frames
overlap, so the title shows the interval, not each frame's span
(core/measurement/frame_times.h).

Architecture documents changed first: logical-overview.md ("The frame":
the two halves, what crosses, the latency), change-axes.md (the queues
are GPU backend, the split and hand-off Frame graph, neither leaks into
the core's schedule or the passes), file-mapping.md (where each part
lives). Rules that decided the design: GPU.7 (queues, events and one
resource per frame in flight, and its caveat that queues alone do not
guarantee overlap), GPU.8 (each barrier and event names its hazard; no
queue drains), GPU.9 (the hand-off made at prepare, never per frame),
MEM.4 (double-buffered data crossing a frame), CP.3 (no copy shared by a
writer and a reader at once), I.4 (`Parts`, a type, not a flag), E.2 and
I.5 (two parts for a one-part graph refused), TLM.11 (timestamps of both
queues compared on the GPU's one clock).

**Measured** on the release build, the demo graph (wide network), the
marbles, per-frame GPU records (two frames in flight, as the window) and
per-dispatch timestamps; GPU idle before every run. On battery: the
machine stayed off AC throughout this step; 4241a5d measured on battery
matches the cost map's AC figure (76.8 against 77 ms), so the battery did
not throttle these runs. One round each; scripts and raw data in
serenity-scratch/async-overlap/.

| Output, time | 4241a5d | two halves | frames a second |
|---|---|---|---|
| 3456 × 2234, t = 120 | 76.8 ms | 73.8 ms | 13.0 → 13.5 |
| 3456 × 2234, t = 20 | 78.5 | 76.8 | 12.7 → 13.0 |
| 1920 × 1080, t = 120 | 23.7 | 21.2 | 42.2 → 47.1 |
| 1920 × 1080, t = 20 | 24.2 | 21.6 | 41.3 → 46.2 |

Frame interval, median. **Latency**: a frame's span on the GPU, light
half's start to image half's end, grows from 76.7 to 146 ms at the
window's size (23.7 to 41.9 at 1920 × 1080): the extra frame. From the
frame's inputs being recorded to its image done, what the window shows
late by, it is 153 against 147 ms (47.3 against 42.2): two frames in
flight already held one frame back, and the shorter interval pays for
the longer span.

**The overlap is real; the hardware is shared.** The timestamps show the
halves concurrent for almost all of each frame (light 74.4 to 148.2 ms
beside image 74.0 to 148.7, one frame). But each kernel slows beside the
other. At the window's size, alone against overlapped:

| | alone | overlapped |
|---|---|---|
| light half (with the full-size albedo) | 49.6 ms | 73.8 |
| image half | 27.5 | 74.5 |
| the denoiser's network | 26.0 | 60.8 |
| spatial reuse | 9.5 | 27.2 |
| tone map | 1.0 | 13.3 |

Run together the two halves take 74 ms where in turn they took 77: the
GPU gains 4% of throughput at the window's size and 10% at 1920 × 1080,
not the 40% the counters suggested (ray tracing 91% busy with arithmetic
at 12 to 15% beside an arithmetic-bound denoiser). Measured: the network
runs at 43% of its own rate beside the light, and the light at 67% of
its own beside the network. Inferred, not yet shown by counters taken
during the overlap: the counters' idle arithmetic is not free to take,
since a core's threads, registers and threadgroup memory are one pool,
and the ray-traced kernels' threads hold their share of it while they
wait on the ray tracing unit. Small dispatches suffer most, queued
behind the other queue's long threadgroups: the depth median 0.55 → 8.9
ms when it ran beside the light's initial sampling. The copy into the
hand-off does not resolve in the timestamps (a stamp after a blit does
not wait for it): 31 MB read and 31 written at the window's size, a
fraction of a millisecond at the memory's bandwidth.

**Kept and rejected.** The full-size albedo first ran in the image half,
where it read as the denoiser's input; there it ran beside the light
half's tracing, both on the ray tracing unit, 8.5 → 31.7 ms, and the
frame got slower than 4241a5d at the window's size (82.7 and 84.3 ms
against 76.8 and 78.5), faster only at 1920 × 1080 (21.4, 21.8). Moved to
the end of the light half, every ray-traced kernel on one queue, it is
the table above: kept. Three frames in flight, to let the light half run
further ahead, was tried as an experiment (scratch only): the image
half's part failed on the GPU with Metal's timeout error
(MTL4CommandQueueErrorDomain 1) at 1920 × 1080 within 20 frames,
reproducibly, and passed under Metal's validation layer, which slows the
frame; not pursued, and two frames in flight stay.

**The image is unchanged.** Every graph in graphs/ against 4241a5d's
build, serenity-headless at 480 × 270 frames 40 and 41 (PNG; PFM for those
that accumulate; --features where there is the features pass), the
denoised graphs at 1920 × 1080 frames 120 and 121 and with time frozen:
88 files, all the same bytes. The hand-off itself is tested where it can
fail (tests/gpu/denoise_test.cpp): each frame's image half held back by a
delay kernel while the next frame's light half runs, every frame as
displayed and the last one's radiance the same bits as the frames run in
turn; a shared radiance copy, a shared feature set or a shared full-size
albedo each fails it.

What it leaves for the next steps: the light half is now the frame
(73.8 ms overlapped) and what slows it is the denoiser beside it, so ray
masks and the split structure, which shorten the light, are measured
against this frame, and the overlap is measured again after each.
