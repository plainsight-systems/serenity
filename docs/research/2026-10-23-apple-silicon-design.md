# The frame on the M3 Max's hardware

Status: the profile (sections 1 to 3) and the design (sections 5 to 7),
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

What the counters say, facts first:

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
  full rebuild" (W22b).
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

## 6. The changes

In the order of what each can move (sections 2 and 3) and what depends on
what. Each its own commit; the demo's frame time before and after at both
sizes; Metal's 94 outputs the same to the bit where the picture should
not change, and a changed picture shown side by side; a quality trade the
user's. Expected gains are stated before measuring.

**1. The intersector, on the boxes as they are.** Every tracing pass
walks candidates with an intersection query and tests each box's shape
in its own loop (`acceleration/trace.metal.h`). The intersector, with the
same exact tests as intersection functions (one a shape kind, not one
function for every kind, as Apple asks), lets the M3's reorder stage
gather the tests into coherent SIMD groups and cuts the ray-tracing
unit's scratch memory, which section 3's occupancy targets are spent on.
The same hits, so the same picture, to the bit if the shapes' tests are
unchanged (checked by the 94 outputs). Expected: the largest single
lever on the tracing passes (32 ms of 62 at 3456 x 2234), size unknown;
it isolates the API from the geometry.

**2. Every shape as opaque triangles on the ray-tracing unit.** Apple's
and Cycles' practice on M3, and the 3090's design (its section 4): each
sphere a mesh at the level whose gap is within a thirtieth of a pixel at
3456 x 2234, the hit point the triangle's (Falcor's), the shading normal
the sphere's; MNEE's seed keeps its box structure and intersection
function under its own mask bit. The earlier test on this machine
(2026-10-09) traced 4 shapes through the query; this is 661 through the
intersector. Three layouts, measured, since Apple warns of instance
overlap and Cycles instances every mesh: Cycles' (a structure a level,
an instance a shape, the top level rebuilt each frame); Apple's
concatenation (the still marbles' placed triangles in one primitive
structure built once and compacted, the fireflies' in another refit each
frame, two instances); and one flat structure of every placed triangle,
refit each frame. The picture changes as any geometry's does (the 3090's:
no facet seen, each channel's mean within 0.01 of a level): side by side.

**3. The network's convolutions.** 21.2 ms, the F32 pipeline 98% busy:
half inputs and float sums already, so fewer multiplies or another
pipeline is what moves it, not occupancy alone. Measured on the network's
own layer shapes in a benchmark before the pass changes, the candidates
in production's order:
- Apple's own: Metal Performance Primitives' `convolution2d` (or
  `matmul2d` as the implicit product), half in, float sums, in the
  shader, so in the frame's own command buffer; OIDN's choice of Apple's
  kernels is MPSGraph's, which encodes outside Metal 4's command buffers.
- Winograd's F(2x2, 3x3) on the simdgroup matrices: 2.25 times fewer
  multiplies for a 3 x 3 convolution, as MPS, cuDNN and ncnn use it.
- Half sums (the 3090's change 8): the work moved to the F16 pipeline,
  half the accumulators' registers; the network's sums change, the user's
  call.
Each against the PyTorch fixture's tolerance (`tests/gpu/denoise_test.cpp`)
and then the frame side by side; any change to the sums is the user's
decision.

**4. The full-size albedo by rasterization.** 8.1 ms of a 97% busy
ray-tracing unit, one ray an output pixel to find the first surface that
is not glass or a mirror. Apple's hybrid design rasterizes that surface
(W21); with change 2's meshes the first surface is a raster pass at the
output's size, and only pixels whose first surface is glass or a mirror
trace on from it (primary surface replacement, NVIDIA's, as the features
pass does now). After change 2, measured first: if the traced albedo is
already cheap there, this is not built. No production example of a
rasterized first surface with primary surface replacement was found; the
parts are each production's.

**5. ReSTIR PT's work over lists, as on the 3090.** MNEE's walks in rounds
(2.6 ms; the 3090's: 2.6 to 1.0) and both reuses over lists of pairs, the
reconnections as well as the replays (7.1 ms of shifts and replays,
reconnected inline now). The same pictures to the bit but for MNEE's
float atomics, as on CUDA.

**6. Threadgroups.** Every per-pixel pass dispatches 1,024 threads a
threadgroup, the maximum (`device/support.cpp`); Apple asks for the
smallest multiple of the SIMD width that fits the work, Cycles uses 64 on
the M3. Swept once, 64 to 1,024, the frame's time the measure. The same
picture.

**7. RTXDI's reservoir layout and the fused shade**, as on CUDA: the
array passes are bound by address translation and the last-level cache
(section 3), what the 16 x 16 blocks address. The same picture.

## 7. Not taken, with why

- MPSGraph for the network, OIDN's way: it encodes into Metal's earlier
  command buffers, not Metal 4's, so the frame would split across queues
  with events between; Metal Performance Primitives are the same vendor's
  kernels inside the frame's own encoder (change 3).
- MetalFX's denoised upscaler (Apple family 9): a different network and
  its own inputs (roughness, specular albedo); replacing the trained
  network is not this milestone's.
- A wavefront integrator, Cycles' design: measured losing on the 3090
  (`2026-10-21-tracing-divergence.md`, section 8.6); the M3 reorders
  intersection functions in hardware, which change 1 tries first.

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
- C2: Blender 4.0 release notes, Cycles — https://developer.blender.org/docs/release_notes/4.0/cycles/
- O1: Intel Open Image Denoise, `devices/metal/` (`metal_conv.mm`, `metal_device.mm`), github.com/RenderKit/oidn at 6602ee2 — https://github.com/RenderKit/oidn
