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

## Step 2: ray masks

Built 2026-10-17 (core/contracts/ray_mask.h, contract 19;
src/metal/acceleration/scene_acceleration.h, "Ray masks"). Every shape is
in one class, a light (a firefly) or a surface, and every ray carries a
mask: **everything**, for a ray that asks what it reaches, or the
**occluders**, the surfaces alone, for a ray that asks whether a chosen
light is visible. A firefly casts no shadow (logical-overview.md,
principle 11).

| Ray | Mask |
|---|---|
| camera rays, the path's bounces (the path tracer, ReSTIR PT's initial paths), ReSTIR DI's BSDF candidate, a shift's primary hit and replay, the features' rays, the preview's camera and reflected rays | everything |
| every shadow ray toward a light sample (path tracer, ReSTIR DI, ReSTIR PT and its shifts' reconnection to a light), a shift's reconnection check between two vertices, MNEE's seed segment and its solved chain's segments, the preview's test of the sky | occluders |

**How a mask is kept.** Metal, like DXR and Vulkan, tests masks only on
instances, and the scene is one level of boxes with no instances
(2026-10-09-acceleration-structure.md). So on that level each mask is a
structure: everything is the structure as it was, and the occluders a
second primitive structure over the 37 surfaces, its boxes naming their
shapes through a table, built once at start-up since no surface in the
marbles moves (per frame slot when one does). Each pass binds one block
holding both and the table (structure_block.h), where it bound the one
structure. The values are mask bits, so Step 3's two-level structure can
put each class's bit on its instance and pass the ray's mask to the query
unchanged; which mask each ray carries does not move.

**The literal alternative, measured and rejected:** Metal's own instance
masks, one bottom level per class under a top level of two instances,
each carrying its class's bit (scratch, not committed). It renders the
same image as the committed design, bit for bit, and is slower than no
masks at all, because every ray, the camera's included, now crosses the
instance level. One round, 3456 × 2234, t = 120 (this round ran on a
noisier machine than the three below; the order held at every size):

| | no masks | two structures (kept) | instance masks |
|---|---|---|---|
| frame interval | 74.9 ms | 71.7 | 87.0 |
| light half, alone | 50.1 | 46.6 | 62.3 |
| features + full-size albedo (camera rays only) | 11.2 | 11.1 | 15.9 |

**Measured** on the release build, the demo graph (wide network), the
marbles, AC power, the GPU idle before every run (no serenity window,
utilization at most 5% for 6 s), 3 rounds interleaved with c84ff47, the
median of the rounds' medians. Frame interval as the window runs (two
frames in flight, the halves overlapping); per-dispatch timestamps with
frames run one at a time, so each pass is its own time alone. Scripts and
raw data: serenity-scratch/ray-masks/ (campaign.sh, table.py, out/).

| Output, time | c84ff47 | masks | frames a second |
|---|---|---|---|
| 3456 × 2234, t = 120 | 73.7 ms | 70.5 | 13.6 → 14.2 |
| 3456 × 2234, t = 20 | 75.1 | 72.0 | 13.3 → 13.9 |
| 1920 × 1080, t = 120 | 21.1 | 20.4 | 47.3 → 48.9 |
| 1920 × 1080, t = 20 | 21.5 | 20.8 | 46.5 → 48.1 |

Each pass alone, ms, 3456 × 2234 (1920 × 1080 in brackets), t = 120:

| Pass | c84ff47 | masks |
|---|---|---|
| initial sampling | 15.6 (5.0) | 14.4 (4.7) |
| MNEE solve | 4.5 (1.5) | 4.6 (1.5) |
| temporal reuse | 7.3 (2.2) | 6.3 (2.0) |
| spatial reuse | 9.4 (2.6) | 8.0 (2.3) |
| features, full-size albedo | 11.1 (3.8) | 11.0 (3.8) |
| light half | 49.5 (15.7) | 45.9 (14.8) |
| image half | 27.4 (8.0) | 27.4 (8.0) |

t = 20 within 1% of the same: light half 51.2 → 47.6 (16.1 → 15.2). The
passes whose rays are visibility rays get faster, the shifts most (13 to
15%, where a reconnection's shadow ray is most of the tracing); the
passes that trace only camera rays do not change, as they should not;
MNEE's solve is 2% slower (inferred, not shown: its segments are few and
its cost arithmetic, and each candidate now reads the table). The light half is 7% shorter
alone and the frame 4% (3% at 1920 × 1080), since overlapped the frame
is held by both halves sharing the GPU (Step 1).

**The image.** What changes is the light the fireflies' own shadows took
away: one firefly between a surface and another, a firefly perched on
the table shading the wood beside it. One frame of the demo, unmasked
and masked, the same frame index, at 1920 × 1080: displayed MSE
**4.1 × 10^-5 at t = 20** and **1.9 × 10^-5 at t = 120** (8-bit display
values in [0, 1], over every pixel and channel). The frames side by side
(unmasked left) and the difference amplified 8 times are in the session's
scratchpad, m4-masks/demo-1080-t20-* and demo-1080-t120-*. Read from the
difference image, not measured apart: most of it is the reuse's noise
taking another course once a few visibility tests differ, spread over
the lit table and marbles, with brighter rims under fireflies resting on
the table.

Which outputs change, by the identity set of Step 1 at a lit frame
(frames 1200 and 1201, t = 20) against c84ff47: every graph that traces
a shadow ray, the path tracer among them; not the test pattern, nor any
feature image (albedo, normal, depth, motion), which are camera rays
alone, byte for byte. With every ray made to carry everything (scratch),
all 88 files of the set are the same bytes as c84ff47's: the change in
the image is the masks' choice and nothing else in the plumbing.

Tested where it can fail (tests/gpu/acceleration_test.cpp): a ray
carrying the occluders is never stopped or blocked by a light and finds
the surface beyond as its own shape at its own distance, while one
carrying everything still finds the light; both masks keep up with
moving shapes, slot by slot; and the two edges, a scene of surfaces alone
(the occluders are everything) and of lights alone (they find nothing).

**Not kept yet:** committed for the user's call on the image.

## Decisions after Step 2 (the user, 2026-10-17)

- **Ray masks kept**: 3 to 4% on the frame for the fireflies' own
  shadows, which are not visible.
- **Step 3, the split scene, skipped on Metal.** Its point, building the
  still scene once, is already met (the occluders' structure is built once,
  and the per-frame rebuild of the one-level structure costs about 0 ms),
  and the standard two-level structure measured slower here because every
  ray pays for the instance level (Step 2's table; 2026-10-09-acceleration-
  structure.md). It is kept for the Vulkan and Xbox milestones, where
  two-level structures are the hardware's normal case.
- **Next: Step 4, checkerboard rendering.**

## Step 4: checkerboard

Built 2026-10-17 (core/contracts/checkerboard.h, contract 20;
src/metal/passes/checkerboard_fill/checkerboard_fill.h;
src/metal/passes/restir_pt/restir_pt.metal.h, "Checkerboard rendering";
src/metal/frame/renderer.h, "Checkerboard rendering"). A setting of the
frame graph, `checkerboard = true`, only beside ReSTIR PT's passes and the
denoise pass; off unless written, and not set in any graph in graphs/
(the demo graph gets it if the user keeps it).

**What production does, read in its source.** RTXDI's checkerboard mode
(RTXDI-Library: Rtxdi/Utils/Checkerboard.hlsli, ReservoirAddressing.hlsli,
Rtxdi/PT/TemporalResampling.hlsli and SpatialResampling.hlsli,
Source/RtxdiUtils.cpp and ReSTIRDI.cpp; the FullSample's
LightingPasses.cpp, PT/FinalShading.hlsl and CompositingPass.hlsl): frame
n traces the pixels whose x + y has the parity its field gives, the
fields alternating each frame; every lighting dispatch is half as wide,
its thread mapped to the field's pixel; the reservoir buffers are half
width, pixels 2k and 2k + 1 of a row sharing slot k; a temporal candidate
is moved into the previous frame's field by one pixel along the row and
searched at radius 2, a spatial one into the current field likewise; the
G-buffer stays whole, every pixel every frame; the missing half of the
lighting is NRD's to rebuild in its checkerboard mode (with no denoiser
the sample shows the stale buffer). Console checkerboard rendering
(Frostbite's and Rainbow Six Siege's resolves) rebuilds a missing pixel
from the previous frame's samples where its motion and its depth or ID say
the surface is the same, from the current frame's neighbors where not.

**What is built**, the same:

- ReSTIR PT's initial sampling (with it the caustics' requests and so
  MNEE's solve), temporal and spatial reuse, shade and duplication map run
  one thread a traced pixel over the packed grid, their arrays packed as
  RTXDI's reservoirs are; the light tiles, the frame's own, do not change.
- The temporal input: RTXDI's, each try at twice the ring's reach and
  moved into the frame before's field.
- The spatial partner: the paired reuse taken over the traced field's
  lattice, so it stays an involution (RTXDI's move into the field would
  break the pairing, one shift no longer serving both ends), its textures
  built at sigma over the lattice's step, sigma pixels as without it.
- The duplication map over the packed grid: 17 x 17 slots, the same 289
  reservoirs, over 34 x 17 pixels.
- The features at the light size stay whole, as RTXDI's G-buffer does:
  they are the denoiser's guide images and the fill's evidence of whether
  a moved pixel is the same surface. **A departure from the brief**, which
  listed them among the halved passes: halving them would save 1.3 ms of
  the 2.7 they take at the window's size and leave the network's edges and
  the fill's test to guesses.
- The fill, at the end of the light half where the hand-off's copy was:
  the traced pixels copied; each untraced pixel from the light the frame
  before traced where its motion says its surface was (the two pixels of
  that frame's field in the bilinear footprint, kept where the depth seen
  there is the depth the motion predicts, within 2% plus the surface's
  depth slope over a pixel's jitter), else from the traced pixels beside it
  of the same depth. Its history is its own, by the frame's parity (62 MB at
  the window's light size), so a readback between frames (the headless
  renderer's PFM) does not take the frame before away.

**Kept and rejected in the fill**, on the marbles at t = 20, 40 frames,
the light handed to the denoiser (scratch, serenity-scratch/checkerboard/):
the bilinear reprojection of the frame before's whole handed-off image (as
SVGF and TAA) took fills back in, a tap of almost no weight carrying a
neighbor's fill wherever the pixel's own failed the test, fills of fills
compounding: the untraced half 8% darker than the traced. A normal test
beside the depth's (within 60 degrees): 3.5% darker, since the features'
normal is of the surface seen through glass, and a sample that reached a
firefly through a marble is both bright and has a firefly's normal. As
built: the untraced half 2.8% darker than the traced, and the whole 0.9%
darker than every pixel traced, the rest being the fireflies seen directly,
a pixel or two across and moving, which the frame before's field often did
not hold. The light itself, before the fill, is the same in mean (the
traced half 0.3049 against 0.3023 with every pixel traced, within the
40-frame mean's noise). A depth test of 2% alone failed at the far table,
seen at a grazing angle and out of focus: 11% of the untraced pixels fell
back to their neighbors, against 4.6% with the slope, then mostly at the
marbles' blurred silhouettes (both counted beside the normal test).

**Measured** on the release build, the demo graph against the demo graph
with `checkerboard = true`, the marbles, AC power, the GPU idle before
every run, 3 rounds interleaved with 706d32b, the median of the rounds'
medians; frame interval as the window runs (two frames in flight), and
per-dispatch timestamps with frames run one at a time. Scripts and raw
data: serenity-scratch/checkerboard/ (campaign_cb.sh, table.py, out/).

| Output, time | 706d32b | checkerboard | frames a second |
|---|---|---|---|
| 3456 × 2234, t = 120 | 71.7 ms | 56.0 | 13.9 → 17.8 |
| 3456 × 2234, t = 20 | 73.2 | 56.9 | 13.7 → 17.6 |
| 1920 × 1080, t = 120 | 20.7 | 16.7 | 48.3 → 59.9 |
| 1920 × 1080, t = 20 | 21.1 | 16.9 | 47.5 → 59.1 |

Each pass alone, ms, 3456 × 2234 (1920 × 1080 in brackets), t = 120:

| Pass | 706d32b | checkerboard |
|---|---|---|
| initial sampling | 14.6 (4.7) | 8.1 (3.0) |
| MNEE solve | 4.6 (1.5) | 2.6 (1.0) |
| temporal reuse | 6.4 (2.0) | 3.5 (1.2) |
| spatial reuse | 8.1 (2.3) | 4.5 (1.4) |
| shade, duplication | 1.3 (0.4) | 0.8 (0.2) |
| features, light size | 2.7 (1.2) | 2.7 (1.2) |
| full-size albedo | 8.4 (2.6) | 8.5 (2.6) |
| the fill (the copy before it: unresolved) | — | 0.6 (0.2) |
| light half | 46.5 (15.0) | 31.6 (11.0) |
| image half | 27.9 (8.1) | 27.9 (8.1) |

t = 20 within 4% of the same: light half 48.2 → 32.6 (15.4 → 11.3). The
halved passes take 56% of their time at the window's size and 63% at
1920 × 1080, where a dispatch's fixed costs weigh more. The light half
is 32% shorter alone and the frame 22% (19% at 1920 × 1080), the
halves overlapping (Step 1); at 1920 × 1080 the frame is 16.7 ms, 60
frames a second. Latency, a frame's span on the GPU: 134 → 112 ms at the
window's size, 41 → 33 at 1920 × 1080.

**The image.** The denoised demo at 1920 × 1080, without and with the
checkerboard, the same frames (t = 20 and t = 120, 31 frames of history
each), side by side, with 2× crops and the difference amplified 8 times,
and a 4-second clip of each at 30 fps from t = 20 (consecutive frames,
H.264 at 16 Mb/s as the Milestone 3 films), alone, side by side, and
cropped to the marbles at 2×: the session's scratchpad, m4-checkerboard/.
Displayed MSE between the two **7.0 × 10^-4 at t = 20, 3.3 × 10^-4 at
t = 120** (one frame to the next without the checkerboard: 2.6 and
3.0 × 10^-3). What looks different, from the crops: the checkerboard's
image is smoother, with less of the speckle on the table and at the
marbles' silhouettes, and a little softer: the striped marbles' edges,
the fireflies' sparkle in the steel marble and the caustics under the
foreground glass are less crisp and slightly dimmer. Inferred, not
measured: the network sees a light half of it a frame old, and filters it
harder than the full frames it was trained on; the light handed to it
differs from the full frame's by the fireflies alone (above). The
displayed mean is the same (0.379 against 0.380 at t = 20, 0.368 both at
t = 120); the linear output is 6% brighter over 40 frames at t = 20, all
of it in the fireflies' cores above white, which the tone map rolls off.
In motion: the mean change from one frame to the next is the same
(0.0224 against 0.0225 of display range over the clip), and the light
size's checker pattern is not in the output (its component 6 × 10^-5 of
display range, 3 × 10^-5 without). Whether the network should be
retrained on checkerboard input is the user's call; not retrained.

**Off, the image is unchanged.** Every graph in graphs/ against 706d32b's
build, the identity set of Step 1: all 88 files the same bytes, in the
final build. One earlier run of it differed in one file (the denoised
graph with time frozen, frame 7) in one pixel by one unit in the last
place, and that case was the same bytes in four repeats with either
build: the MNEE light's float atomics, whose order the hardware chooses
(contract 16 states it).

Tested where it can fail: every pixel is traced in exactly one of any two
frames in a row, across the index's wrap; the packed grid takes each
traced pixel once, at its own slot, and no other; the frame before's pixel
beside a pixel is one it traced (tests/checkerboard_test.cpp); each
frame's lattice pairing is an involution over the traced field, every
partner traced, sigma pixels apart (tests/gpu/restir_pt_test.cpp); and the
setting's refusals.

**Not kept yet:** committed off, for the user's call on the image and on
whether the network needs retraining on checkerboard input (not
retrained).
