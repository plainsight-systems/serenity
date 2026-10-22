# The frame on the RTX 3090 Ti's hardware

Status: design, 2026-10-20, for the user's review before any header or
code. The CUDA backend (`2026-10-19-cuda-backend.md`) ported Metal's
design: "Every pass is the port's, untuned" (its Phase 5, 11.2 ms at
1920 x 1080). This note starts from the other end, as the user asked: what
the GA102 accelerates, then where the frame's work belongs on it, then the
changes, each with what guards it. CUDA becomes Milestone 5 when this is
done (the user, 2026-10-20).

## 1. The hardware

From the machine (a device query of CUDA's attributes and OptiX's device
properties, 2026-10-20) and NVIDIA's documents. "Machine" means the 3090
Ti answered it; a letter is the source (end of the note).

| Unit | Fact | Source |
|---|---|---|
| SMs | 84, sm_86, 1.86 GHz boost | machine |
| SM partitions | 4, each with a warp scheduler, a 64 KB register file, 16 FP32 lanes and 16 lanes that do FP32 or INT32: 128 FP32 a clock, or 64 FP32 + 64 INT32 | G |
| Threads | at most 1,536 (48 warps) and 16 blocks an SM; 255 registers a thread; 65,536 registers an SM | machine, T |
| L1 / shared | 128 KB an SM together; shared at most 100 KB an SM, 99 KB a block (above 48 KB by opt-in), carveouts 0, 8, 16, 32, 64, 100 KB | machine, G, T |
| L2 | 6 MB; up to 4.1 MB can be set aside to persist (access-policy windows) | machine, B |
| Memory | 24 GB GDDR6X, 384 bits at 10,501 MHz: 1,008 GB/s (21 Gb/s a pin x 48 bytes) | machine (bandwidth computed) |
| RT cores, 2nd generation (OptiX: RT core version 20) | in hardware: BVH traversal, ray against box, **ray against triangle** (twice Turing's rate), instance transforms; shading runs beside them | machine, G (Table 4) |
| Not in hardware | spheres, curves and custom primitives: traversal reaches their boxes in hardware, then each candidate leaves the RT core for an intersection program on the SM. OptiX's built-in spheres are such a program (hardware spheres first come with Blackwell) | F, BW |
| Reordering | none: OptiX reports shader execution reordering 0 on this GPU; `optixReorder` is a no-op on Ampere, and the guide's advice for it is "a legacy path that performs manual shader binning" | machine, O |
| Instance masks | 8 bits | machine, O |
| Tensor cores, 3rd generation | `mma.sync` f16 inputs. On GeForce, f32 accumulation runs at **half** the f16-accumulation rate (RTX 3090: 71 against 142 dense TFLOPS; the same die as RTX A6000 has no halving). On this card, about 80 and 160 (the 3090's figures scaled by SMs and clock, not NVIDIA's) | G (Table 9) |
| FP64 | 1/64 of FP32 | G |
| Copy engines | 2; concurrent kernels | machine |

## 2. The frame on it today

As it was when this was designed, before Phase 1 (section 7 says what
each phase changed).

The demo graph on marbles at 1920 x 1080 (read from the code and one
trace's launch records, 2026-10-20). The light half runs at 960 x 540 and
traces a 480 x 540 checkerboard; the image half runs at 960 x 540 and
1920 x 1080.

**The rays.** Six of the light half's passes trace (the image half
traces none): ReSTIR PT's initial pass (a path
of up to 9 surfaces, a shadow ray or an MNEE seed at each), MNEE's solve
(a Newton walk of up to 64 iterations through up to 4 interfaces), the
temporal and spatial shifts and replays, and the features twice (up to 9
traces a pixel). **No ray meets a triangle.** The scene's shapes are 661
spheres (45 marbles and cores, 616 fireflies) and 6 boxes, all custom
primitives. So the RT cores traverse, and every candidate a ray meets
leaves them for the SM's intersection program and comes back, which is
the round trip the hardware has no shortcut for. The main structure of
all 667 shapes is rebuilt each frame, because the fireflies move. A ray
leaves a surface from a point offset by its own rounding (Wächter and
Binder's, since 2026-10-11, `integrator/surface.cuh`); the intersection
program also skips a shape named in payload 0, which only shadow rays
pass, and they carry the occluders, which hold no light. There
is no instance level, and the two questions (every shape, the occluders
alone) are two structures with default build flags and no compaction.

**The SMs.** Every tracing pass is one OptiX program that loops over its
path's vertices: the rays of one warp go to different materials (glass,
coated, metal, the swirled cores), and with no hardware reordering they
diverge for the rest of the path. MNEE's solve and the replays run
from lists, launched at the lists' capacity (518,400 and 1,555,200
threads) whatever the count, because an OptiX launch's size is the
host's. Registers of the plain kernels (from the launch records):
spatial merge 101 a thread (16 warps an SM, a third of the 48: computed
from the registers and its blocks of 8 warps, not read), the
denoiser's output 87, its convolutions 43 to 70. The OptiX programs' are
not in the launch records; reading them needs Nsight Compute, which needs
the GPU's counters permission (a root setting).

**The tensor cores.** The denoiser is 74.5 GFLOP a frame (eleven 3 x 3
convolutions on five levels, from 960 x 544 down, and the 1 x 1 head;
computed from the network's layout). It runs on `mma.sync` m16n8k8 with
f32 accumulation, which on this card has a floor of about 0.93 ms at the
peak rate; f16 accumulation would halve the floor. Its loads into shared
memory are plain and single-buffered; its tiles are 4 warps of 2 x 16
pixels.

**Memory.** ReSTIR PT reads and writes about 1.3 KB a traced pixel a
frame (the pass's own tally), 337 MB, 0.33 ms at the full bandwidth: not
where the frame's 11 ms goes, unless the accesses are scattered (32-byte
sectors). The reservoirs are 64 bytes, as RTXDI's ReSTIR PT reservoir
is, in row order. The scene's arrays are read from global memory without
the read-only path's hints.

**Launches.** 8 OptiX launches and 49 kernels a frame, on two streams; no
CUDA graph; 7 OptiX pipelines.

## 3. What shipped systems do on this generation

- **NVIDIA's OptiX guide:** triangles alone on the RT cores; for a scene
  of nothing but triangles, `usesPrimitiveTypeFlags` = triangles. Fast
  trace builds and compaction for structures that don't change. One
  geometry structure where each mesh has one instance; an instance level
  for many instances of one mesh, its transforms accelerated by the RT
  core. On GPUs without reordering, manual binning. Cooperative vectors
  (the tensor cores from a ray program) are supported on Ampere, best up
  to 64 wide. (O)
- **RTXDI (NVIDIA's ReSTIR):** reservoirs stored block-linear, 16 x 16
  pixels contiguous, so a spatial neighborhood is a few cache lines; a
  checkerboard mode that halves the reservoir buffer; DI reservoirs 24
  bytes, GI 32, PT 64 (R).
- **Blender Cycles on CUDA/OptiX:** sorts paths by shader before shading
  (a counting sort, in partitions of 65,536 paths for locality); hit
  objects through `optixTraverse`, as ours (C).
- **OIDN's CUDA backend:** CUTLASS implicit-GEMM convolutions on
  m16n8k16, NHWC, block tiles 256 x 32 x 32 or 256 x 64 x 32 with three
  pipelined stages, f16 accumulation where its "fast math" is asked for
  (D). tiny-cuda-nn keeps a layer's weights in registers and activations
  in shared memory (N).
- **Wavefront path tracing** (Laine et al. 2013, Kepler): queues per
  material made speed independent of the number of materials; not
  repeated on Ampere in any source found (L).

## 4. The changes, from the hardware

Each row: the unit it moves work to or the cost it removes, whether the
picture stays the same to the bit, and so what guards it. Byte identity
here is CUDA's own outputs on the 3090 before and after (the Mac's are
untouched unless the core changes).

| # | Change | Hardware reason | Picture | Guard |
|---|---|---|---|---|
| 1 | **Boxes as 12 triangles** | ray against triangle on the RT core, no round trip | the same box; t and the point in the last bits | before/after frames |
| 2 | **One instance level**: a geometry structure per kind of shape (the unit box; the sphere, below), the 667 shapes its instances, the mask bits answering the two questions (every shape; occluders alone) in one structure | instance transforms on the RT core; a frame rebuilds 667 instances instead of every shape's boxes; one structure instead of two | with change 4 | before/after frames |
| 3 | **Build flags and compaction**: fast trace, compacted, for what doesn't move | the guide's | the same | byte identity |
| 4 | **The spheres as triangles** (below): icosphere meshes, a level a sphere by its error, instanced | ray against triangle on the RT core for 661 of 667 shapes | changes: a mesh within a thirtieth of a pixel | before/after frames |
| 5 | **Fused passes**: ReSTIR PT's shade into the spatial merge (both a pixel's, the merge's reservoir read back at once); the depth of field's and the tone map's per-pixel chains (6 and 12 kernels) into fewer | a pass's output read from registers, not from memory; fewer launch gaps | the same | byte identity |
| 6 | **Lists by the machine's size**: MNEE's solve and the replays as a grid of 84 x 1,536 threads striding over the list, not the list's capacity | no launch of a million threads that return | the same, except MNEE's: its float atomics already make its sums order-dependent (QUEUE.md) | byte identity for the replays; frames for MNEE |
| 7 | **The denoiser on the tensor cores' own shape**: m16n8k16, pipelined loads into shared memory (`cp.async`, double-buffered), tiles sized so the five levels fill 84 SMs (wave quantization) | half the instructions a product; loads beside the math; no idle SMs on the small levels | the same sums, perhaps in another order | the PyTorch fixture's tolerance, then frames |
| 8 | **f16 accumulation in the denoiser** (the user's decision, below) | twice the tensor rate on GeForce | changes the network's sums | frames, the user's (below) |
| 9 | **Reservoirs block-linear** (16 x 16, RTXDI's) and **sized to the checkerboard** (half the slots today) | a spatial neighborhood in a few sectors of L2; half the memory | the same | byte identity |
| 10 | **The frame as a CUDA graph**, the scene's arrays through the read-only path, one pipeline for the tracing passes | launch gaps; the unified L1/texture cache | the same | byte identity |

**The spheres as triangles.** Ampere intersects no sphere in hardware,
so a sphere on the RT core is a triangle mesh: production practice, and
how NVIDIA's Marbles at Night was made (the user, 2026-10-20). An
icosphere's vertices lie on the sphere and its faces inside it; its gap,
the deepest a face lies below the sphere, is a fraction of the radius
that falls by four a level (computed from the meshes,
`cuda/acceleration/shape_meshes.h`; about twice an even edge's sagitta,
an icosphere's triangles not all alike):

| Level | Triangles | Gap / radius | Nearest marble, 3456 x 2234 | The 0.75 m brass sphere, 3456 x 2234 |
|---|---|---|---|---|
| 3 | 1,280 | 4.5e-3 | 1.0 px | 3.2 px |
| 5 | 20,480 | 2.8e-4 | 0.063 px | 0.20 px |
| 6 | 81,920 | 7.1e-5 | 0.016 px | 0.050 px |
| 7 | 327,680 | 1.8e-5 | 0.004 px | 0.013 px |

So each sphere takes **the coarsest level whose gap is within a
thirtieth of a pixel** at the largest size the project renders (3456 x
2234, the Mac's window), seen from the nearest the camera comes to it:
the camera's place grown by its lens, from outside the sphere or within
it (a cavity), a moving sphere's from its motion's extent
(`animation/extent.h`). A mesh is built for each level used, once,
compacted, every sphere an instance of its level's; levels 0 to 9, a
sphere no level serves (its surface can reach the lens) refused by name
(`cuda/acceleration/mesh_levels.h`). The marbles scene takes two
meshes: level 5 for 649 spheres (the fireflies and most marbles), level
6 for 12, 102,400 triangles; the brass scenes levels 4 to 7, the brass
sphere level 7.

- **The point** is the triangle's: the hit's vertices and barycentrics,
  placed by its instance, as Falcor and RTXDI find a hit's point
  (`shapes/shapes.cuh`, `acceleration/trace.cuh`). On the surface the
  ray traced to a few ulps of its own coordinates, which is what the ray
  offset assumes (2026-10-11), wherever the sphere is and whichever way
  the ray leaves: the leave test holds, no ray re-hitting the surface it
  left, at every scale it tries (a 3 mm firefly to a 20 m sphere, the
  origin to 10 km off).
- **The normals:** the geometric normal is the triangle's, so a ray
  leaves on the side of the surface it traced; the shading normal is the
  sphere's own at the point's place on it, as a smooth mesh's
  interpolated normal stands for it, so the BSDF sees the sphere. The two
  differ by under a degree at these levels; a sample above the shading
  plane and below the triangle's, at a grazing angle, leaves on the
  triangle's far side, as pbrt-v4 and Falcor let it (both offset by the
  geometric normal, neither drops it). A texture's object-space point is
  the sphere's.
- **MNEE** solves on the sphere itself (its own roots,
  `integrator/mnee.cuh`); its chain's visibility, traced, meets the mesh
  within its tolerance (a thousandth of the radius). Its seeds are found
  by an intersection program over the occluders' boxes
  (`integrator/mnee_seed.cuh`), which needs every sphere a segment
  enters, and tests them exactly: those boxes are a structure of their
  own, an instance in the same instance level under a mask bit only seed
  rays carry.
- **The shadow rays' ignored shape:** a shadow ray carries the occluders,
  which hold no light, so the light it aims at is never among what it can
  meet, and a triangle needs no program to skip it.
- **Considered, not taken:**
  - the hit point put back on the sphere, as the custom primitives had
    it, with a level fine enough that the mesh clears the ray offset
    inward. Tried and removed: the offset is about 1.5e-5 of a point's
    coordinates and the gap a fraction of the radius, so a glass sphere
    centred near the origin, its coordinates about its radius, needs a
    gap under 3.8e-6 of it, past level 8; the test scenes' glass spheres
    (0.5 to 1 m at the origin) were refused. The triangle's point removes
    the bound rather than tightening it;
  - the spheres kept as custom primitives (every candidate's test on the
    SM); OptiX's built-in sphere (NVIDIA's program, still on the SM on
    Ampere); a triangle shell refined to the exact sphere (a second
    mechanism for a difference the levels put below a thirtieth of a
    pixel); one mesh for every sphere (the fireflies would descend a
    fine tree for a shape 1.5 mm across).

The picture changes from today's, as any change of geometry does: frames
before and after, side by side.

**The decision for the user: f16 accumulation.** It doubles the denoiser's tensor rate
on this card and changes its sums; OIDN takes it only under "fast math".
Proposed: measured and shown after change 7, the user's call.

## 5. Phases

In the order of the work each moves and what depends on what. No
measurement says which matters most (the user's rule: no breakdown
first); the order is an inference from the units: every tracing pass
runs through the structure, so it is first.

1. **The structure on the RT cores**: changes 1, 2, 3 and 4, measured
   as one (the four cannot be separated without code written to be
   thrown away: an instanced structure of custom boxes). Done
   2026-10-20; results in section 7.
2. **Fewer passes and idle threads**: changes 5 and 6. Narrowed by the
   user 2026-10-21 after the passes' own times (section 7): change 6 and
   the depth median's selection by the block; change 5's fusions not
   made (they hold under 0.6 ms, most of it beside the light half). Done
   2026-10-21: change 6 tried and removed, the selection kept.
3. **The tensor cores**: change 7, then change 8 by the user's decision.
4. **Bytes and launches**: changes 9 and 10.

Each phase: its headers with both guideline corpora's rule IDs, then the
code; the demo's frame headless (`serenity-headless --times`, release,
the GPU idle, no window), before and after, at 1920 x 1080 and 3456 x
2234; a pass's own GPU timestamps only for a change aimed at that pass.
Gates once at a phase's end, on the 3090 (and the Mac when the core
changes).

**Rollback.** Each change is its own commit, on the work branch; Metal
untouched unless the core changes, its 94 outputs then the guard.

## 6. Not decided here

- Reading the OptiX programs' registers needs Nsight Compute with the
  counters permission (root). Asked of the user when a phase targets a
  program's occupancy, not before.
- **Binning by material without hardware reordering.** Cycles sorts
  paths by shader (a counting sort) because its integrator is a
  wavefront: a trace, a sort, a shade, the next vertex. Ours is one OptiX
  program a path, so binning needs that split first: a design of its own
  (the path tree as queues, Laine et al.'s), considered after these.
- **RTXDI's fused resampling** (temporal and spatial in one pass) changes
  the estimator, not only the passes: a picture change and a ReSTIR
  design question, not this machine's tuning.
- Cooperative vectors (the network inside a ray program): considered
  after these.

## 7. Results

**Phase 1, the structure on the RT cores** (2026-10-20). Release build,
the GPU otherwise idle, headless `--times` over frames 7000 to 7599 (t =
117 to 127 s), each period's mean steady; before is the work branch at
ba7105d, built and run the same hour:

| Size | Before | After |
|---|---|---|
| 1920 x 1080 | 11.3 ms | 10.15 ms (-10%) |
| 3456 x 2234 | 39.6 ms | 35.9 ms (-9%) |

The instance level and the seed's boxes built for fast building instead
of fast tracing: the same, 10.7 and 38.2 ms against 10.7 and 38.3 over
the run's first 600 frames, so fast tracing stays, the guide's. The
picture: the demo frame at t = 120, before and after, shown to the user
side by side: the same picture, no facet seen, each channel's mean
within 0.01 of a level, the grain different as the paths are.

**The meshes' bound, measured** (2026-10-21, the user's call). The
thirtieth of a pixel was chosen in the design, not asked; production
tessellates to about a pixel (Cycles' default dicing rate, Nanite's
target). Three builds differing only in the bound, the demo graph over
frames 7000 to 7599, and the frame at t = 120 at 3456 x 2234 beside each
other:

| Bound | Marbles' levels | 1920 x 1080 | 3456 x 2234 | The picture |
|---|---|---|---|---|
| 1/30 px | 5, 6 | 99.1 fps (10.09 ms) | 27.7 fps (36.1 ms) | as before the meshes |
| 1/4 px | 3, 4 | 100.7 fps (9.93 ms) | 28.3 fps (35.3 ms) | the frame's mean 1.5% darker |
| 1 px | 2, 3 | 101.8 fps (9.82 ms) | 28.6 fps (35.0 ms) | 4% darker; the porcelain marble dark, the swirled cores dull through their glass |

No facet shows on any outline, the brass sphere's included, at any of
the three. What changes is the light: a coarse mesh's triangles tilt
several degrees from the sphere's own normal, which the BSDF shades by,
so more grazing samples leave on the triangle's far side; and a coarse
firefly is smaller than the sphere its light is sampled on (inferred
from the frames, not measured apart). At a thirtieth of a pixel both
are under a degree, and the frame's mean matched the frame before the
meshes to 0.01 of a level. **Kept: a thirtieth of a pixel**, the user's
decision 2026-10-21: the coarser bounds buy 3% at most and change the
picture.

**Phase 2** (2026-10-21). Before it, each pass's GPU time on the demo
at 1920 x 1080 (an Nsight Systems trace of frames 7020 to 7039, the
passes Phase 2 aimed at, as section 5 allows): the light half, the
frame's critical path, about 8.9 of its 10.1 ms:

| Pass | GPU time |
|---|---|
| ReSTIR PT's initial (path tree, NEE, MNEE's seeds) | 2.65 ms |
| MNEE's solve | 2.57 ms |
| the spatial shift | 1.22 ms |
| the features at the output's size | 0.74 ms |
| the temporal shift | 0.50 ms |
| the features at the light's size | 0.28 ms |
| the temporal and spatial replays | 0.23 and 0.18 ms |
| the instance level's rebuild | 0.14 ms |
| the rest (merges, shade, duplication, fill, light tiles) | about 0.4 ms |

Change 5's fusions would save under 0.6 ms of passes that mostly run
beside the light half (ReSTIR PT's shade, 0.016 ms; the depth of field's
and the tone map's chains); the three tracing programs at the top hold
6.4 ms. So the user narrowed the phase (section 5):

- **Change 6, tried and removed.** The replays and MNEE's solve launched
  over the GPU's resident threads (84 x 1,536), each striding over its
  list: MNEE's solve went from 2.57 to 6.29 ms, the replays from 0.23 and
  0.18 to 0.26 and 0.21 ms, the outputs the same to the bit. A list
  launched at its capacity lets the hardware give a finished warp's place
  to the next; a fixed set of threads striding over requests of very
  different lengths (MNEE's Newton walks) waits on each warp's slowest
  chain (GDSA.8). The threads that return at once cost less than that.
- **The depth median's selection by the block, kept.** One thread walked
  up to 2,048 bins a level, 0.21 ms in all on one SM of 84; the block's
  scan (CUB's) takes 0.011 ms, the median the same to the bit. On the
  image half, beside the light half, so the frame does not show it: 98.3
  to 98.8 fps (10.17 to 10.12 ms) at 1920 x 1080 and 27.7 to 27.8 fps
  (36.1 to 35.95 ms) at 3456 x 2234, within a run's spread; the 94
  outputs the same as Phase 1's.

The time is in the three tracing programs: their design is next
(section 6, "Binning by material").

What it found on the way: the hit point put back on the sphere needs the
mesh to clear the ray offset inward, a bound no level meets for a glass
sphere near the origin (section 4, "Considered, not taken"); the
triangle's point, Falcor's, removed it. The tests that held a hit to the
sphere's exact distance take each backend's gap from its harness
(Metal's 0); the leave test counts a self-hit on the part of a shape a
ray left, a mesh's triangle, and sees each shape from where the scenes
see one, so its mesh is the scenes' level.

## 8. Phases 3 and 4, designed (2026-10-21)

Taken up by the user 2026-10-21 after the tracing passes' work
(`2026-10-21-tracing-divergence.md`, section 8): changes 7, 5, 9 and 10,
in that order; change 8 stays the user's call.

**Where the frame is.** The demo at 1920 x 1080, 168.6 fps (5.93 ms): the
light half 4.15 ms of GPU work, the image half 2.26 ms, run on their two
streams (frame N's image half beside frame N + 1's light half). Each
frame takes about 11 ms from start to end and one ends every 5.9 ms, so
the halves overlap, and the GPU is full: the frame is the two halves'
work, 6.4 ms, less 0.5 ms the overlap recovers. Measured by the
submission's own events (a probe build). Nsight Systems' CUDA trace
shows the halves one after the other (a frame there takes 6.6 ms): its
tracing serializes the streams, so a trace's timeline is not the frame's
schedule, only each kernel's time. Every millisecond either half sheds
is the frame's.

### Change 7: the convolutions on the tensor cores' own shape

What limits them (Nsight Compute, `conv3x3_n4` at 1920 x 1080): the
tensor pipe busy 34 to 41% of the time; the warps stalled most on the
tensor pipe's issue rate (math pipe throttle), then on the block's
barriers between copying a block of 8 input channels and computing on it,
then on the copy's global loads; 4.4 million of the 8.0 million shared
memory wavefronts bank conflicts. NVIDIA's own answer on Ampere is
CUTLASS's implicit-GEMM convolution (`cutlass/conv`, the Ampere tensor-op
mainloop): mma.sync m16n8k16, ldmatrix from conflict-free shared layouts,
cp.async copies several stages ahead. Taken in two steps, each measured
and kept only if it wins:

1. **m16n8k16 and conflict-free tiles.** A K of 16 input channels a
   product, two 8-channel blocks, half the products' instructions;
   A by ldmatrix.x4 (16 pixels by 16 channels), B by ldmatrix.x2.trans; a
   width of input channels that is an odd number of blocks (the test
   network's 8, 24, 40) takes its last block by m16n8k8. The halo held
   as one plane a block of 8 channels, a pixel's 16 bytes a row, so
   ldmatrix's 8 rows are 128 contiguous bytes; the weights' rows padded
   by 16 bytes, so B's 8 rows fall on distinct banks. The results
   written from the accumulators' fragments, two channels a lane, no
   staging through shared memory. The tile stays contract 18's (4 rows
   of 32 pixels a block, up to 32 output channels).
2. **The copy ahead of the math.** cp.async (16 bytes, zero-filled past
   the level's border) into a second stage while the first is multiplied,
   CUTLASS's multistage mainloop at two stages: the barrier stall and the
   loads' latency off the math's path, for twice the shared memory (36 KB
   a block at 32 output channels, 2 blocks an SM where 7 run now).

**The picture.** The tensor core sums 16 products before adding them to
the accumulator, so the sums round in another order: not the same to the
bit. The guard is the PyTorch fixture (`tests/gpu/denoise_test.cpp`, its
tolerance), then the demo's frame side by side.

**Change 7, measured** (2026-10-21). Each convolution kernel's GPU time
a frame (Nsight Systems, the demo at 1920 x 1080, frames 7000 on), the
variants built from one source, two constants apart (K, stages), every
one with the 16-byte copies by cp.async, the padded taps and the results
from the fragments:

| Variant | 32 outputs a block | 24 | 16 | Shared memory, registers (32) |
|---|---|---|---|---|
| before: K 8, ushort4 loads, staged results | 914 us | 293 us | 87 us | 12 KB, 70 |
| K 8, one stage | 820 | 277 | 75 | 9 KB, 70 |
| **K 16, one stage** | **784** | **260** | **73** | 18 KB, 85 |
| K 8, two stages | 813 | 270 | 72 | 18 KB, 80 |
| K 16, two stages | 942 | 282 | 77 | 36 KB, 75 |

Kept: K 16, one stage, 177 us a frame fewer (14%). The second stage,
CUTLASS's multistage mainloop, lost: at this tile (4 warps, 32 x 4
pixels, 32 outputs) its shared memory leaves 2 blocks an SM, where CUTLASS
pays for its stages with larger tiles. A first build lost to every
variant until its accumulators were indexed by loops the compiler did not
unroll: 128 bytes of each thread's accumulators in local memory (the
compiler's report, STACK:128); every loop over the tile unrolled
(`#pragma unroll`), none.

The demo, against 2022573: 168.6 to 172.7 fps (5.93 to 5.79 ms) at
1920 x 1080, 48.5 to 49.4 fps (20.63 to 20.25 ms) at 3456 x 2234. The
94 outputs: the denoised graphs' 11 changed, every other the same; the
demo's frame side by side, 0.3% of its pixels by one or two levels, the
means the same.

### Changes 5, 9 and 10, from what production does and the frame's own launches

The frame's operations, from the latest trace (7,020 frames, the demo at
1920 x 1080 after change 7): a frame launches 59 kernels, 8 OptiX
programs, 13 memsets and 18 copies. The light half: the 8 OptiX launches
and about 18 kernels. The image half: about 41 kernels (the denoiser's
27 with the depth median's 6, the depth of field's 6, the tone map's 12),
no OptiX launch. What each change does, from the shipped system it
follows, with what is expected before it is measured:

**Change 5, ReSTIR PT's shade into the spatial merge.** RTXDI's fused
mode shades a pixel in the dispatch that resampled it. Here the shade reads
only the light buffer, and the schedule puts it straight after the spatial
pass whenever that pass is present (core/frame/schedule.h). So the merge,
having added its estimate to the pixel's light, folds that light into
the accumulated image itself (the shade's own code, `accumulate` and
`count_non_finite`), and no shade launches. The same additions in the
same order: the same to the bit. A graph without the spatial pass
launches the shade as now; Metal keeps its own passes. Expected: the
shade's 16 us and its launch, under a run's spread in frame time.
Measured (2026-10-22, against change 7, each build twice, alternated):
172.7 to 172.9 fps (5.79 to 5.785 ms) at 1920 x 1080, 49.6 to 49.7 fps
(20.15 to 20.11 ms) at 3456 x 2234, within a run's spread; the 94
outputs the same.

**Change 5, the depth of field's and the tone map's chains: kept as
passes.** Section 4 called them per-pixel chains; they are not. Each
kernel reads a neighbourhood of the one before: the tiles' dilation, the
gather over presorted neighbours, the median, the bloom's 13-tap
downsample and tent upsample. UE5's depth of field (its flatten,
dilate, setup, reduce, gather, postfilter and recombine) and its bloom,
like Call of Duty: Advanced Warfare's, keep each as its own pass. What the
fusion was for, the launch gaps between small kernels (the tone map's 10
level kernels hold 42 us of work), is what change 10's graph removes, the
passes kept.

**Change 9, RTXDI's reservoir layout.** RTXDI maps a pixel to its
reservoir in two steps: `RTXDI_PixelPosToReservoirPos` halves x under
the checkerboard, and `RTXDI_ReservoirPositionToPointer` stores the
result in 16 x 16 blocks, each 256 contiguous elements, rows of blocks
`ceil(width / 16) * 256` apart. The first step is contract 20's packing
already (`checkerboard_slot`); what is new:

- **The allocation sized to the packed grid**, `checkerboard_width(w) x
  h` with the checkerboard and `w x h` without, padded to whole 16 x 16
  blocks: the core's `path_reuse_arrays` takes the checkerboard, so both
  backends allocate half the per-pixel arrays (Metal's packing is the
  same contract; its row-linear slots fit the padded size). Memory only:
  no time expected from it.
- **The slots block-linear on CUDA**: `Traced::slot` and its inverse
  `slot_pixel` (passes/restir_pt/restir_pt.cuh) RTXDI's mapping, and the
  duplication pass's own row-linear index through it. A per-pixel launch
  is already 16 x 16 threads (`per_pixel_block`), so a block's threads
  write one contiguous 256-slot span, and a neighbour within the spatial
  disc (radius 30) lies a few blocks away, not up to 30 rows of 960
  slots. Every array indexed by slot follows (reservoirs, primaries,
  light, shifts, replays, finals, previous positions and pixels,
  duplication); the pair lists' capacity is the slot count times the
  neighbours. The same to the bit. Metal keeps its row-linear slots: the
  layout is this GPU's (L2 sectors and DRAM pages), not the contract's.
  Expected: within the pairs' and merges' 0.4 ms, a few percent of it.

**Change 10, the image half as a CUDA graph; the light half's runs of
kernels too.** OptiX's launches cannot be captured. Tried on the 3090
(OptiX 9.1, driver 595.91): a captured frame instantiates and launches,
but its OptiX launches are not in the graph, only their parameter
copies, and the frame faults. An NVIDIA engineer's answer on the
developer forum says the same. So:

- **The image half, whole**: its 41 kernels recorded under stream
  capture on the image stream, after the hand-off wait (which stays
  outside the graph, as the frame's begin and completion events do, so
  `--times` reads the same events), and launched as one graph. The graph
  is captured again each frame and the slot's executable graph updated in
  place (`cuGraphExecUpdate`), instantiated again only when its shape
  changes (a new size, the history's first frame). This is the CUDA
  programming guide's whole-graph update, and what llama.cpp's CUDA
  backend does each token. So whatever `record_image` records is what
  runs, and no parameter is baked in at a capture.
- **MNEE's walk** (its start, 7 rounds and settle: 9 kernels between
  OptiX launches) captured the same way, in its pass. The light half's
  other runs are one to three operations, where a graph's own launch
  costs what it saves.
- **An OptiX launch under capture is refused**: `TracedProgram` and
  `ShiftReplay` ask `cuStreamIsCapturing` and throw, since the driver
  would otherwise drop the launch without an error.
- Expected: the image half's launch gaps, perhaps 40 to 80 us of GPU time
  a frame, which the light half partly fills already. Up to 1%.

**Change 10, read-only loads: production already.** Cycles reads its
scene arrays through const pointers in its launch parameters with plain
loads (`kernel_data_fetch` in `kernel/device/{cuda,optix}/globals.h`: no
`__ldg`), as this renderer does (cuda/scene/scene_block.cuh). On sm_86
global loads are cached in L1, which is the same cache as the texture
path (T). No change.

**One pipeline for the tracing passes: withdrawn.** It came before the
register rule (2022573). An OptiX pipeline launches as one kernel, so
the programs sharing it would share one register allocation, the largest
limit (the tree's 200), undoing the 128 the shifts, the features and the
path tracer run at (inferred from what a launch is, not measured).

Each change is its own commit, the 94 outputs the same to the bit for
each, the demo's frame time before and after; the gates once, after
change 10.

## Sources

- G: NVIDIA Ampere GA102 GPU Architecture whitepaper, v2 —
  https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
- T: NVIDIA Ampere GPU Architecture Tuning Guide —
  https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- B: CUDA C++ Best Practices Guide —
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- O: OptiX 9 Programming Guide —
  https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.251219.LTR.pdf
- F: NVIDIA developer forum, "OptiX 7 spheres" (2023) —
  https://forums.developer.nvidia.com/t/optix-7-spheres/259057
- BW: NVIDIA RTX Blackwell GPU Architecture —
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- R: RTXDI —
  https://github.com/NVIDIA-RTX/RTXDI-Library (reservoir layouts and
  addressing), https://github.com/NVIDIA-RTX/RTXDI (Doc/RestirPT.md)
- C: Blender Cycles —
  https://github.com/blender/blender, intern/cycles/integrator/path_trace_work_gpu.cpp
- D: OIDN —
  https://github.com/RenderKit/oidn, devices/cuda/cutlass_conv_sm80.cu
- N: tiny-cuda-nn — https://github.com/NVlabs/tiny-cuda-nn
- RTG6: Wächter, Binder, "A Fast and Robust Method for Avoiding
  Self-Intersection", Ray Tracing Gems (2019), ch. 6 —
  https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_6.pdf
- L: Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013 —
  https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf
