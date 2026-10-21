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
leaving a surface ignores it by name: the intersection program skips the
shape in payload 0, so no ray meets the surface it starts on. There
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
| 1 | **Boxes as 12 triangles** | ray against triangle on the RT core, no round trip | changes: a triangle has no intersection program, so the skip by name goes (below, "Leaving a surface") | before/after frames |
| 2 | **One instance level**: a geometry structure per kind of shape (the unit box; the sphere, below), the 667 shapes its instances, the mask bits answering the two questions (every shape; occluders alone) in one structure | instance transforms on the RT core; a frame refits 667 instances instead of rebuilding every shape; one structure instead of two | likely the last bits: the world-to-object transform moves from our program's arithmetic to the RT core's | byte identity if it holds, else frames |
| 3 | **Build flags and compaction**: fast trace, compacted, for what doesn't move | the guide's | the same | byte identity |
| 4 | **The spheres as triangles** (below): one level-5 icosphere mesh, instanced | ray against triangle on the RT core for 661 of 667 shapes | changes: a mesh and the origin offset (below) | before/after frames |
| 5 | **Fused passes**: ReSTIR PT's shade into the spatial merge (both a pixel's, the merge's reservoir read back at once); the depth of field's and the tone map's per-pixel chains (6 and 12 kernels) into fewer | a pass's output read from registers, not from memory; fewer launch gaps | the same | byte identity |
| 6 | **Lists by the machine's size**: MNEE's solve and the replays as a grid of 84 x 1,536 threads striding over the list, not the list's capacity | no launch of a million threads that return | the same, except MNEE's: its float atomics already make its sums order-dependent (QUEUE.md) | byte identity for the replays; frames for MNEE |
| 7 | **The denoiser on the tensor cores' own shape**: m16n8k16, pipelined loads into shared memory (`cp.async`, double-buffered), tiles sized so the five levels fill 84 SMs (wave quantization) | half the instructions a product; loads beside the math; no idle SMs on the small levels | the same sums, perhaps in another order | the PyTorch fixture's tolerance, then frames |
| 8 | **f16 accumulation in the denoiser** (the user's decision, below) | twice the tensor rate on GeForce | changes the network's sums | frames, the user's (below) |
| 9 | **Reservoirs block-linear** (16 x 16, RTXDI's) and **sized to the checkerboard** (half the slots today) | a spatial neighborhood in a few sectors of L2; half the memory | the same | byte identity |
| 10 | **The frame as a CUDA graph**, the scene's arrays through the read-only path, one pipeline for the tracing passes | launch gaps; the unified L1/texture cache | the same | byte identity |

**The spheres as triangles.** Ampere intersects no sphere in hardware,
so a sphere on the RT core is a triangle mesh: production practice, and
how NVIDIA's Marbles at Night was made (the user, 2026-10-20). Its
tessellation is set by its error on screen, not chosen to fit a time:

| Icosphere level | Triangles | Its edge's error at the nearest, largest marble, 1920 x 1080 | at 3456 x 2234 |
|---|---|---|---|
| 4 | 5,120 | 0.064 px | 0.13 px |
| 5 | 20,480 | 0.016 px | 0.033 px |

(The nearest sphere's surface 0.25 m from the camera, the largest radius
12.5 mm, the camera's 28 degrees; the error is the chord's sagitta, r(1 -
cos(a/2)) for an edge of angle a.) So **level 5**: a thirtieth of a pixel
at the largest size in play. One unit-sphere mesh, a geometry structure
built once, compacted, every sphere an instance of it, so 45 marbles and
cores cost one structure of 20,480 triangles; the fireflies the same
mesh (their own level, by the same bound, if a smaller one serves).

- **The normal** at a hit is the sphere's own there (from its center, in
  the instance's space), as a smooth mesh's interpolated normals stand
  for it: so the shading and MNEE's solve, which works on the sphere
  itself (its own roots, `integrator/mnee.cuh`), see one surface.
- **MNEE's seeds** are found today by an intersection program over the
  casters' boxes (`integrator/mnee_seed.cuh`), which needs every sphere a
  segment enters, not the nearest. It stays as it is: the casters' boxes
  a structure of their own, an instance in the same instance level under
  a mask bit of its own that only seed rays carry.
- **Considered, not taken:** the spheres kept as custom primitives (as
  today: every candidate's test on the SM); OptiX's built-in sphere
  (NVIDIA's program, still on the SM on Ampere); a triangle shell refined
  to the exact sphere (the RT core's candidate, then a quadratic, a
  re-trace for grazing rays): exact, but a second mechanism for a
  difference the tessellation already puts below a thirtieth of a pixel.

**Leaving a surface.** Today a ray ignores the shape it leaves by name,
in the intersection program; a triangle has none, an any-hit program
would run for every candidate (the cost the triangles remove), and 8 mask
bits cannot name 667 shapes. Production practice: the ray's origin moved
off the surface by an offset scaled to the hit point's floating-point
error (Wächter and Binder, "A Fast and Robust Method for Avoiding
Self-Intersection", Ray Tracing Gems ch. 6; what Falcor and RTXDI use).

The picture changes from today's, as any change of geometry does: frames
before and after, side by side, where the glass is thickest.

**The decision for the user: f16 accumulation.** It doubles the denoiser's tensor rate
on this card and changes its sums; OIDN takes it only under "fast math".
Proposed: measured and shown after change 7, the user's call.

## 5. Phases

In the order of the work each moves and what depends on what. No
measurement says which matters most (the user's rule: no breakdown
first); the order is an inference from the units: every tracing pass
runs through the structure, so it is first.

1. **The structure on the RT cores**: changes 1, 2, 3 and 4. Measured
   after each.
2. **Fewer passes and idle threads**: changes 5 and 6.
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
