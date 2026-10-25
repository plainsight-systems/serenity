# The frame on the M3 Max's hardware

Status: the profile (sections 1 to 3) and the design (sections 5 to 7, the
changes revised by the user's direction),
2026-10-23, for the user's review before any header or code (the plan's
steps 0 and 1, docs/process/QUEUE.md, Milestone 6), as the RTX 3090 Ti's
was (`2026-10-20-nvidia-hardware-design.md`). Milestone 4 applied
engines' standard optimizations on this machine; nothing yet is designed
for its own hardware.

## 1. The machine

| | |
|---|---|
| GPU | Apple M3 Max, 40 GPU cores |
| OS, tools | macOS 26.6.2 (25G83), Xcode 26.2, `xctrace` 26.0 (17C52) |
| Power | AC, the GPU idle before each run (no window, wallpaper, video or Photos analysis running) |
| Threads | SIMD groups of 32 (`threadExecutionWidth`); every pipeline reports 1,024 threads a threadgroup (`maxTotalThreadsPerThreadgroup`), the convolutions and the denoiser's output included: with the M3's Dynamic Caching a pipeline's registers no longer bound that number, so occupancy is read from the GPU's counters (section 3), not from the API |

## 2. The frame, and each pass's time

**The frame** (`serenity-headless --times`, the demo graph on the marbles,
frames 7000 to 7599, t = 117 to 127 s, the second of two runs; the first
runs agreed on the mean and had wide ranges):

| Size | Frame |
|---|---|
| 1920 x 1080 | 56.6 fps (17.67 ms) |
| 3456 x 2234 | 15.8 fps (63.21 ms) |

Milestone 4 recorded 57 fps (17.5 ms) and 17.1 fps (58.5 ms) by its own
harness. 1920 x 1080 agrees; at 3456 x 2234 this is 8% slower, not yet
explained (open, section 4).

**The two halves' overlap.** The same frames rendered one at a time, each
waited for before the next is begun (below): 20.0 ms at 1920 x 1080 and
62.1 ms at 3456 x 2234. Overlapping frame N's image half with frame N + 1's
light half saves 2.4 ms at 1920 x 1080 and nothing at 3456 x 2234, where
the overlapped frame is 1 ms longer.

**Each pass's GPU time** (median of 80 frames at t = 120 s after 21
warm-up frames, rendered one at a time; a precise timestamp after every
dispatch, Metal 4's counter heap; the probe is scratch code, never
committed, as Milestone 4's frame-cost map was):

| Block | 1920 x 1080 | 3456 x 2234 | Share at 3456 x 2234 |
|---|---|---|---|
| The denoiser: depth median, input, 11 convolutions, 4 pools, output | 7.77 ms | 26.54 ms | 43% |
| of which the convolutions | 6.08 | 21.24 | 34% |
| ReSTIR PT's initial pass (the power pyramid, the light tiles, the path tree) | 3.36 | 8.95 | 14% |
| The features at the output's size (the full-size albedo) | 2.53 | 8.10 | 13% |
| The spatial reuse: shift, replay, merge | 1.48 | 4.71 | 8% |
| The temporal reuse: shift, replay, merge | 1.27 | 3.69 | 6% |
| The features at the light's size | 1.17 | 2.62 | 4% |
| MNEE's solve | 0.96 | 2.56 | 4% |
| The depth of field | 0.70 | 2.47 | 4% |
| The tone map | 0.31 | 1.06 | 2% |
| The shade, the duplication map, the fill | 0.36 | 1.23 | 2% |
| **The frame, one at a time** | **20.0** | **62.1** | |

## 3. What limits each pass

The GPU's Performance Limiters counters at 3456 x 2234, t = 120 s (the
saved Instruments template through `xctrace`, every dispatch separated by
a one-thread spin so the samples can be cut into dispatches, as Milestone
4's were). Only the larger dispatches; percentages are each counter's mean
inside the dispatch:

| Dispatch | ms | Limited by | Occupancy (target) |
|---|---|---|---|
| the convolutions, each of the 11 | 0.16 to 7.03 | F32 arithmetic 80 to 98%, F16 0% | 20 to 31% (86 to 100%) |
| the denoiser's output | 3.00 | F32 57% | 20% (100%) |
| ReSTIR PT's initial | 8.76 | ray-tracing unit 78%, last-level cache 39% | 15% (30%) |
| the features at the output's size | 7.99 | ray-tracing unit 97% | 22% (24%) |
| the features at the light's size | 2.58 | ray-tracing unit 86% | 20% (30%) |
| the shifts and replays, four | 1.21 to 2.63 | ray-tracing unit 75 to 90% | 16 to 18% (23 to 35%) |
| MNEE's solve | 2.60 | F32 56%, instruction issue 50%, ray-tracing unit 33% | 13% (40%) |
| the merges, the shade, the fill, the denoiser's input | 0.21 to 1.15 | memory: address translation (MMU) 38 to 81%, last-level cache 33 to 61% | 31 to 88% |
| the depth of field, the tone map | 0.19 to 1.03 | instruction issue 33 to 96% | 33 to 99% |

The 1920 x 1080 recording did not save: on this machine `xctrace` saves a
counters recording once or twice after its services start and fails from
then on, idle runs included (exit 138, an empty trace; an Instruments left
running since 2026-10-10 had blocked every save before it was quit). The
display's size is the hard case the measurement rule asks for; its
timestamps at 1920 x 1080 are in section 2.

What the counters say, facts first (occupancy is reported against its
target, but a pass whose pipeline is saturated is bound by that pipeline,
not by occupancy; the convolutions below are such):

- **Fact:** the network's convolutions run entirely in 32-bit arithmetic
  (the F32 limiter at 80 to 98%, the F16 limiter at 0), at about a quarter
  of the occupancy the GPU's occupancy manager aims for. **Inference:**
  their registers hold them there; half-precision inputs, which the 3090
  runs with 32-bit sums, would move the work to 16-bit arithmetic and
  halve its registers. The largest block, 43% of the frame.
- **Fact:** every pass that traces is bound by the ray-tracing unit (75 to
  97% active), and the occupancy manager holds those passes to a target of
  23 to 40%. **Inference:** what moves them is fewer ray-tracing cycles a
  ray (the shapes: section 4) or fewer rays.
- **Fact:** the full-size albedo, one ray an output pixel only to find the
  first surface that is not glass or a mirror, costs 8.1 ms, nearly the
  path tree's 9.0, with the unit 97% busy. **Inference:** production
  engines find that surface by rasterizing it, and trace only where glass
  or a mirror sends the ray on (primary surface replacement); a lead the
  3090 did not show, where it costs 0.74 ms.
- **Fact:** the small passes that read and write ReSTIR PT's arrays are
  bound by address translation and the last-level cache. **Inference:**
  RTXDI's 16 x 16 reservoir blocks, which the CUDA backend uses and Metal
  does not, are what these passes' locality asks for; they hold 4% of the
  frame together.

## 4. Open

- The 8% at 3456 x 2234 against Milestone 4's recorded frame (section 2).
- Each pipeline's registers, which Metal does not report; Xcode's GPU
  capture and shader profiler do, should a change need them.

## 5. What Apple says, and what production does on Apple silicon

Gathered 2026-10-23; a letter is the source (end of the note).

**The hardware, as Apple describes the M3 (Apple family 9):**

- The register file is "now a cache instead of the permanent storage";
  threadgroup, tile and stack memory likewise, all assigned dynamically
  from L1, backed by the last-level cache (TT1). An occupancy manager
  sets a target on the threads a core runs, "to keep the registers,
  threadgroup, tile, and stack memory on-chip" (TT2): the targets of
  section 3.
- "FP16 math instructions execute at peak throughput", use fewer
  registers, and convert "at no cost"; the FP16, FP32 and integer
  pipelines run "in parallel to a greater degree than ever before", but
  only for instructions from several SIMD groups (TT1). Apple does not say
  FP16 runs at twice FP32's rate on the M3.
- Ray tracing: "the implementation of the intersector object is
  hardware-accelerated" on Apple family 9, each traversal independent in
  fixed-function hardware; intersection functions still run on the shader
  cores, gathered into coherent SIMD groups by a reorder stage (TT1).
- Simdgroup matrices are 8 x 8; half x half + float compiles to one
  intrinsic (checked with this toolchain). Metal 4's tensor operations
  (Metal Performance Primitives: `matmul2d`, `convolution2d` in shaders)
  run on M3 as "optimized shader implementations"; only the M5's neural
  accelerators run them in hardware (TT3, W25a).
- Threadgroups: "use the smallest multiple of the SIMD width that maps
  well to your workload"; larger ones can leave cores unbalanced (W22a).
  The MMU limiter means address translation, its remedy "reduce your
  memory span and maximize data locality" (W22a, TT2).

**Apple's guidance for this frame's work:**

- "use the intersector object API whenever possible": the intersection
  query "increases the amount of ray trace scratch memory" and "disables
  the reorder stage" (TT1). "Maximize opaque triangle tests, and only use
  custom intersection functions on geometries that need them" (TT2); an
  opaque triangle geometry skips the intersection function entirely (D1).
- Each instance a ray enters costs a transform and a switch (W22b); where
  instances overlap, "concatenate these instances together into a single
  primitive acceleration structure" (TT2). Refit "is much faster than a
  full rebuild", for "deforming or animated models such as skinned
  characters", primitive structures; the instance structure is rebuilt
  whole each frame: "Doing a full rebuild is fine in this case since
  there's only one instance acceleration structure and it usually only
  contains at most a few thousand objects" (W22b).
- Keep the ray payload small: lower latency, possibly higher occupancy
  (TT1).
- Primary visibility: Apple's hybrid design rasterizes the G-buffer,
  which "plays the role of our primary rays", and traces the rest (W21);
  no Apple measurement of it on the M3 was found.

**Production on Apple silicon:**

- **Blender Cycles' Metal backend** (Apple's engineers among its authors):
  on M3 and later it traces with MetalRT by default ("Cycles' own
  intersection code" stays the default on M1 and M2), one triangle
  acceleration structure a mesh, instanced under one top level, static
  scenes built and compacted, dynamic ones refit; points (spheres) as
  boxes with an intersection function; curves as Metal's own curve
  primitives. Every kernel's threadgroup on M3 is 64 threads, "Peak
  occupancy is achieved through Dynamic Caching on M3 GPUs". Its
  integrator is a wavefront (C1, C2).
- **Intel Open Image Denoise's Metal backend** runs its U-Net through
  Apple's MPSGraph, a graph a layer (convolution, bias, ReLU, pooling and
  upsampling fused), with half-precision tensors and weights; its own
  kernels only prepare the input and the output (O1).

## 6. The changes: production on Apple silicon, taken whole

Revised 2026-10-23 at the user's direction: the first draft gated
production's designs on measurements ("only if still costly"), put in a
quick change that is not Apple's practice (Winograd), and ruled out a
layout by estimate. Each change below is built whole; each is measured,
and the measurement says what it bought, not whether it is built. A
change to the picture is shown side by side for the user (the meshes,
the network's 16-bit sums, the raster G-buffer); Metal's 94 outputs guard
every change that should not change it.

1. **Tracing as Blender Cycles traces on the M3** (C1, TT1, TT2): the
   intersector API in every tracing pass, in place of the intersection
   query; opaque triangle geometry Metal intersects in hardware, the
   still spheres (the marbles' pile, whose instances overlap) placed in
   the world and concatenated into one structure a mask, built once and
   compacted, as Apple asks of overlapping instances (TT2; added
   2026-10-24, after change 1's first slices were measured without it:
   the passes of incoherent rays slowed most), the boxes and the moving
   shapes (the fireflies) instanced meshes, the boxes because a box's
   two triangles a face placed in the world carry their far corners'
   rounding to every point of the face, past what the ray offset covers
   (the leave test's ground 400 m across), their top level rebuilt whole each
   frame, as Apple advises for an instance structure (W22b; corrected
   2026-10-24 from "refit", taken from Cycles' viewport updates and
   checked against Apple's session at the user's direction); every
   sphere a mesh at the 3090's levels (a thirtieth of a pixel at 3456 x
   2234), the hit point the triangle's, the shading normal the sphere's;
   MNEE's seed its own box structure with an intersection function,
   one traversal a segment visiting every box it enters, as the CUDA
   backend's (Apple: intersection functions "only on geometries that
   need them"); Cycles' walk of nearest hits (C3:
   `kernel_path_mnee_sample`, at most 10, each a traversal from the
   top) was built first and measured 3.3 ms slower at 3456 x 2234 on the
   marbles, whose segments cross piles of them, and is replaced (decided
   with the user 2026-10-25); small ray payloads, which the intersector
   gives by itself (a hit is its result's distance, ids, barycentrics
   and per-primitive data, nothing carried through an intersection
   function); each ReSTIR PT pipeline specialized by function constants
   carrying the run's path reuse settings, as Cycles' are by its
   KernelData (Cycles' feature flags have no counterpart: the demo scene
   uses every feature its kernels have, so they would fold nothing); and
   64-thread groups, as Cycles' on the M3, a per-pixel dispatch's 8 x 8.
2. **The first surface rasterized, Apple's hybrid design** (W21): a
   render pass draws change 1's meshes into the output-size G-buffer
   (albedo, normal, depth, motion), and only pixels whose first surface
   is glass or a mirror trace on from it (primary surface replacement,
   NVIDIA's, as the features pass does now); the light-size features the
   same at the light's size.
3. **The network on Apple's own kernels, as Open Image Denoise runs it**
   (O1, W25a, TT3): the U-Net through Metal 4's machine-learning path in
   the frame's own command buffers, 16-bit tensors and weights; the
   export to its package added to `training/` (Core ML's tools, pinned
   there); the kernel-predicting head and the output kernel ours.
4. **ReSTIR PT's work over lists, as on the 3090:** MNEE's walks in
   rounds; both reuses over lists of pairs, reconnections and replays.
5. **RTXDI's reservoir layout and the fused shade**, as on CUDA.

## 7. Not built, and one question for the user

- MPSGraph, Open Image Denoise's own route to Apple's kernels: it encodes
  into Metal's earlier command buffers; change 3 reaches the same
  vendor's kernels inside Metal 4's.
- MetalFX's denoised upscaler: another network with other inputs;
  replacing the trained network is not this milestone's.
- **For the user: Cycles' wavefront.** Cycles' integrator on the M3 is a
  wavefront (C1), so production taken whole includes it. On the 3090 it
  was built, measured losing (140.8 against 163.7 fps) and reverted by
  the user's decision; the M3 differs in that its hardware reorders
  intersection functions (change 1). Built after changes 1 to 5 if the
  user wants it measured on this machine.

## Sources

- TT1: Apple, "Explore GPU advancements in M3 and A17 Pro" (tech talk 111375) — https://developer.apple.com/videos/play/tech-talks/111375/
- TT2: Apple, "Discover new Metal profiling tools for M3 and A17 Pro" (tech talk 111374) — https://developer.apple.com/videos/play/tech-talks/111374/
- TT3: Apple, "Accelerate your machine learning workloads with the M5 and A19 GPUs" (tech talk 111432) — https://developer.apple.com/videos/play/tech-talks/111432
- W21: Apple, "Explore hybrid rendering with Metal ray tracing" (WWDC21 10150) — https://developer.apple.com/videos/play/wwdc2021/10150/
- W22a: Apple, "Scale compute workloads across Apple GPUs" (WWDC22 10159) — https://developer.apple.com/videos/play/wwdc2022/10159/
- W22b: Apple, "Maximize your Metal ray tracing performance" (WWDC22 10105) — https://developer.apple.com/videos/play/wwdc2022/10105/
- W25a: Apple, "Combine Metal 4 machine learning and graphics" (WWDC25 262) — https://developer.apple.com/videos/play/wwdc2025/262/
- D1: Apple, `MTLAccelerationStructureGeometryDescriptor.opaque` — https://developer.apple.com/documentation/metal/mtlaccelerationstructuregeometrydescriptor/opaque
- C1: Blender Cycles, `src/device/metal/` (`bvh.mm`, `kernel.mm`, `device.mm`), github.com/blender/cycles at a456b76 — https://github.com/blender/cycles
- C3: Blender Cycles, `src/kernel/integrator/mnee.h` (`kernel_path_mnee_sample`'s probe loop, `MNEE_MAX_INTERSECTION_COUNT` 10), github.com/blender/cycles at a456b76 — https://github.com/blender/cycles/blob/a456b76/src/kernel/integrator/mnee.h
- C2: Blender 4.0 release notes, Cycles — https://developer.blender.org/docs/release_notes/4.0/cycles/
- O1: Intel Open Image Denoise, `devices/metal/` (`metal_conv.mm`, `metal_device.mm`), github.com/RenderKit/oidn at 6602ee2 — https://github.com/RenderKit/oidn
