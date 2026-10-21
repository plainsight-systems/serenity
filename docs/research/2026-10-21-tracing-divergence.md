# The tracing programs' divergence on the RTX 3090 Ti

Status: design, 2026-10-21, for the user's review before any header or
code. The hardware design's Phase 2 (`2026-10-20-nvidia-hardware-design.md`,
section 7) measured where the frame's time is: the light half is the
frame's critical path, about 8.9 of its 10.1 ms at 1920 x 1080 (98.8
fps), and three tracing programs hold 6.4 ms of it. Ampere has no shader
execution reordering (OptiX reports 0 on this GPU), so what a warp's
threads do apart they do one after another. This note is how each of the
three is to do less of that, starting with the one measured.

## 1. Where the time is

The demo at 1920 x 1080, each pass's GPU time (an Nsight Systems trace of
frames 7020 to 7039):

| Program | GPU time | What its threads do apart |
|---|---|---|
| ReSTIR PT's initial pass | 2.65 ms | a path of up to 9 vertices, its length set by misses, emitters and roulette; at each, the material's BSDF (rough, dielectric, conductor, coated), a medium or none, NEE by a shadow ray or MNEE's seed |
| MNEE's solve | 2.57 ms | a Newton walk of 1 to 64 iterations through 1 to 4 glass crossings, then the chain's visibility |
| the spatial shift | 1.22 ms | three partners each, each paired or not, reconnecting to a light or a surface, forced or not, a material at each end |

(The code's map: `src/cuda/integrator/path.cuh`, `mnee.cuh`, `shift.cuh`,
`src/cuda/passes/restir_pt/restir_pt.cuh`.)

## 2. MNEE's solve, measured

A throwaway build printed, for one solve in 37, how its Newton walk ended
and after how many iterations (`integrator/mnee.cuh`, newton()), on the
demo at 1920 x 1080, frames 7100 to 7199: 65,688 solves.

| How the walk ended | Solves | Iterations it used | Iterations, median / p90 |
|---|---|---|---|
| converged | 80.3% | 43.0% | 6 / 13 |
| the step halved below the least (1e-4) | 13.6% | 32.9% | 37 / 55 |
| all 64 iterations, not converged | 5.8% | 23.5% | 64 / 64 |
| stuck (no crossing moves) | 0.2% | 0.6% | 41 / 53 |

Crossings: one 30%, two 66%, three 1.5%, four 3.1%.

A fifth of the solves fail, and they spend 57% of the iterations. In a
warp of 32 nearly every warp holds one, and the warp runs until its
longest walk ends: grouping the samples into warps of 32 at random, the
lanes are busy 26% of the warp's iterations (an estimate: the real warps
hold neighbouring pixels' requests, which may share more). That is the
cost, and it is divergence, not work.

What the samples say each remedy would leave, in warp-iterations of the
Newton loop:

| Remedy | Warp-iterations, of now | The picture |
|---|---|---|
| rounds of 8 iterations, the walks still going gathered into full warps between rounds | 29% | the same: each walk's arithmetic as now |
| rounds of 16 | 39% | the same |
| a cap of 15 iterations | 24% | keeps 91.4% of today's converged walks |
| a cap of 20 | 32% | keeps 95.5% |

## 3. What shipped systems and papers do

- **Blender Cycles** runs its GPU integrator as a wavefront: each stage
  its own kernel over a queue of paths, the paths' state in arrays (SoA),
  the next kernel the one with the most paths queued; surfaces are
  sorted by shader before shading (a counting sort, in partitions). MNEE
  is its own kernel: `shade_surface_mnee`, sorted by shader, in Blender
  3.2 to 4.x; `INTERSECT_MNEE`, unsorted, on main. Its Newton walk is the
  one this is translated from: 64 iterations, threshold 1e-3, the step
  halved on failure and doubled on success
  (`intern/cycles/kernel/integrator/mnee.h`).
- **pbrt-v4's GPU integrator**: a wavefront with a queue and a kernel per
  material, no sorting.
- **Falcor's and RTXPT's path tracers**: one ray generation program,
  shader execution reordering on Ada and later; without it, nothing but
  delta paths split into passes of their own.
- **Laine, Karras and Aila 2013** (Kepler): the wavefront 36% to 221%
  faster than the megakernel as materials multiplied; thread use 23% in
  the megakernel against 53%. Even in a kernel of its own, a material
  with a variable loop ran at 71%: splitting by kind does not undo a
  loop's divergence.
- **Hanika et al. 2015 (MNEE)**: "a maximum iteration count of 15 works
  well in practice": 15 to 50 adds under 10% of converged walks.
- **Hong et al. 2025 (NVIDIA)**: Newton's iterations are specular
  manifold sampling's bottleneck; they cap them at 5 in their shifts,
  and leave scheduling the walks on GPUs to future work.
- **Persistent threads**: tried here and removed (Phase 2: a fixed set of
  threads striding over MNEE's list, 2.57 to 6.29 ms), as Aila and
  Laine's addendum found on Fermi: the hardware's own placing of warps
  does better than static striding.

So nothing published regroups Newton's walks on a GPU; the wavefront's
answer to a variable loop is compaction between stages, which is what
rounds are.

## 4. The design: MNEE's walk in rounds

MNEE's solve becomes three steps, the walks still going gathered into
full warps between rounds (GDSA.8: work split by what is left, not one
irregular walk a thread):

1. **Start** (a plain CUDA kernel: nothing before the walk traces a
   ray): each request read, its seed projected onto its spheres and its
   references set, exactly as manifold_light() does now; then the walk's
   state written to a list: the request's index, the points on each
   crossing, each crossing's reference (set_references() fixes it from
   the first points, so it is carried, not recomputed), the step and its
   size, whether the derivatives are to be computed again, the
   iterations so far. A request whose light faces away ends here, as
   now.
2. **Rounds** (a plain CUDA kernel: the walk traces no ray, its
   projections are each sphere's own roots): each walk still going takes
   up to 8 iterations of the same newton() body; one that converges is
   appended to the converged list, one that fails is dropped, one still
   going is appended to the next round's list (warp-aggregated appends,
   `device/append.cuh`, as the replays'). Eight rounds at most (64
   iterations), each launched at the list's capacity, its threads past
   the count returning (Phase 2: cheaper than any thread striding).
3. **Finish** (an OptiX launch): each converged walk's light, as
   manifold_light() finishes it now: the transfer matrix, the chain's
   transmittance, its visibility (the rays), the light added to its
   pixel.

The state a walk carries between rounds: four points (12 floats), four
references (12), four steps (8), the step size, the flag and the count,
and the request's index: about 150 bytes, two lists of them at the
capacity (one request a pixel, 518,400), 160 MB, made once (MEM.9).

**The picture.** Each walk does the same arithmetic in the same order as
now, so each solve's light is the same number. Only the order in which
the solves add their light to a pixel changes, which float atomics
already make order-dependent (QUEUE.md, Next 0b): the graphs without
MNEE stay the same to the bit; the MNEE graphs' frames are shown before
and after, and `tests/gpu/mnee_test.cpp` (MNEE converging to the path
tracer's caustic) holds.

**What it should save**, and what is not known: the samples put the
walk's warp-iterations at 29% of now. How much of the solve's 2.57 ms is
the walk, and how much the start and the finish (the visibility rays),
is not measured; the per-pass trace after the change says. The rounds'
launches and the state's 150 bytes a walk a round are new costs.

**Considered, not taken here:**

- **A cap on the iterations** (15 or 20, Hanika's and Hong's practice):
  it would cut the work further, but it drops converged walks, a quality
  trade and the user's call. Measured beside the rounds, after them, on
  the caustic where the glass is thickest, if the user wants it.
- **Sorting the requests by crossings** before the walk: the crossings
  set an iteration's cost, not whether the walk fails, which is what
  costs.
- **Persistent threads** taking walks from a global counter: Phase 2's
  static version lost 2.4x; the dynamic one adds an atomic a walk, which
  NVIDIA's staff advise against in OptiX programs, and a block cannot
  choose its SM in an OptiX launch.

## 5. The initial pass and the spatial shift

Their divergence is in material, path length and the shift's outcomes,
not one loop. Measured 2026-10-21 by Nsight Compute (the counters opened
by the user), the eight OptiX launches of frame 7000 at 1920 x 1080:

| Program | Registers a thread | Lanes busy an instruction | Warps active, of the SM's peak | GPU time |
|---|---|---|---|---|
| ReSTIR PT's initial pass | 128 | 21.3 of 32 (67%) | 27% | 2.75 ms |
| MNEE's solve | 128 | 5.8 of 32 (18%) | 14% | 2.74 ms |
| the temporal shift | 128 | 25.9 of 32 | 32% | 0.52 ms |
| the temporal replay | 128 | 18.9 of 32 | 25% | 0.26 ms |
| the spatial shift | 128 | 16.0 of 32 (50%) | 28% | 1.28 ms |
| the spatial replay | 128 | 20.2 of 32 | 28% | 0.20 ms |
| the features, light and output sizes | 116 | 27.1 and 28.8 of 32 | 26 and 29% | 0.30 and 0.78 ms |

- MNEE's lanes are busy 18% of its instructions, below the samples'
  estimate (section 2: the estimate left out the threads past the list's
  count and the work around the walk): section 4 stands.
- The spatial shift's lanes are idle half the time, the initial pass's a
  third: their divergence is real, smaller than MNEE's, and spread over
  materials and outcomes, where a wavefront split with sorting by
  material (Cycles', pbrt-v4's) is the production answer and the larger
  change. Designed after section 4's result.
- **Every tracing program holds 128 registers a thread**, so an SM holds
  16 warps of its 48 (65,536 registers), a third, and the warps active
  sit at that ceiling (25 to 32%). OptiX chooses the count where the
  module leaves it at its default
  (`OptixModuleCompileOptions::maxRegisterCount`, 0); a lower limit lets
  more warps hide latency at the cost of spilling to local memory
  (GPU.3: occupancy as latency-hiding budget, measured, not a score). A
  lever of its own, measured after section 4.

## 6. Steps

1. MNEE's walk in rounds (section 4). Measured: the demo's frame,
   before and after, at both sizes, in frames a second; the solve's
   three steps' GPU times. Gates; the 94 outputs against
   `/data/serenity/identity/phase1.sha` for the graphs without MNEE.
2. With the user's leave for the counters: the initial pass's and the
   spatial shift's SIMT efficiency and registers, then their design.

## 7. Results

**MNEE's walk in rounds** (2026-10-21, section 4 as designed). The demo,
release, the GPU otherwise idle, headless `--times` over frames 7000 to
7599; before is the work branch at 91ead49, built and run the same hour:

| Size | Before | After |
|---|---|---|
| 1920 x 1080 | 99.3 fps (10.07 ms) | **112.7 fps (8.87 ms)**, +13.5% |
| 3456 x 2234 | 27.7 fps (36.07 ms) | **33.4 fps (29.95 ms)**, +20% |

The frames' spread narrowed with it: 8.2 to 9.4 ms a frame at 1920 x
1080, from 6.3 to 14.1. The solve's launches (an Nsight Systems trace,
frames 7020 to 7039): the start 0.18 ms, seven rounds of 0.10 ms each,
the finish 0.09 ms, 0.97 ms in all against the single solve's 2.57; the
light half 7.4 ms of the frame, from 8.9.

The picture: the 94 outputs at 640 x 360 the same to the bit as Phase
1's; the demo's frame at t = 120 at 1920 x 1080 differs in 1.3% of its
pixels, nearly all by one level of 255 (430 by more than two, at most
15), each channel's mean the same to 0.001 of a level: the order in
which the solves add their light to a pixel, float atomics' (section
4). Shown to the user side by side.

**Tried and removed:** the start and the rounds launched over the GPU's
resident threads (84 x 1,536), each thread striding over its list, since
a round's walks take at most 8 iterations: 9.19 ms a frame at 1920 x
1080 against 8.87, the same at 3456 x 2234. A round's 0.1 ms is its
walks', not its launch's: the fifth of the walks that fail go on through
every round.

**The programs' register limit** (2026-10-21, section 5's lever). OptiX,
asked for its compile report (the context's log at level 4), gives each
program's registers and the bytes it spills to local memory: at its own
limit of 128 every tracing program spilled, the initial pass most.

| Program | 128 registers: direct spills | continuation spills | 160 registers: registers, direct spills |
|---|---|---|---|
| ReSTIR PT's initial pass | 540 B | 1,762 B | 160, 200 B |
| MNEE's finish | 204 B | 744 B | 160, 60 B |
| the temporal shift | 156 B | 3,788 B | 155, 0 |
| the temporal replay | 64 B | 910 B | 154, 0 |
| the spatial shift | 264 B | 3,933 B | 160, 0 |
| the spatial replay | 52 B | 910 B | 153, 0 |
| the features | 0 | 510 B | 119, 0 |
| MNEE's seed (intersection) | 20 B | 0 | 108, 20 B |

The continuation spills, the values held across a trace call, move by a
few bytes at most at any limit from 64 to 255: the limit trades only the
direct spills against the warps an SM holds (12 of its 48 at 160, 16 at
128). The demo, as above, one limit for every program:

| Limit | 1920 x 1080 |
|---|---|
| 64 | 75.4 fps (13.26 ms) |
| 96 | 107.5 fps (9.30 ms) |
| 128 (OptiX's) | 112.7 fps (8.87 ms) |
| **160** | **117.0 fps (8.55 ms)**, +3.8% |
| 192 | 115.6 fps (8.65 ms) |
| 255 | 116.1 fps (8.61 ms) |

At 3456 x 2234, 128 against 160, twice each: 33.3 fps (29.99 ms) against
**34.8 fps (28.72 ms)**, +4.4%. Kept at 160 (`cuda/device/pipeline.h`,
max_registers), one limit for every program: the differences between
160, 192 and 255 are a tenth of a millisecond, too small to tune each
program to. Cycles leaves the limit at OptiX's choice; here the spills
measured otherwise (GPU.3). The 94 outputs the same to the bit.

The compile report also warns of double precision in every program
with a sine or cosine: CUDA's sinf and cosf reduce an argument beyond
about 10^5 radians in doubles (Payne and Hanek's reduction). No angle
the programs take comes near it, so the branch is never run; left as
CUDA ships it.

## 8. The initial pass and the spatial shift: measured, then designed

Status: design, 2026-10-21, for the user's review before any header or
code. Section 5's promise: their design after the rounds' result.

### 8.1 Where their lanes are idle

Nsight Compute's source counters (the programs built with line
information, a throwaway build) on frame 7000 at 1920 x 1080, at 160
registers: the initial pass 2.38 ms with its lanes 67.3% busy, the
spatial shift 1.20 ms at 48.0%. The idle lanes are spread over all the
code, not gathered in the materials: in the initial pass the four
materials' own code (Fresnel, dielectric, conductor, coated) runs with 5
to 11 lanes of 32, but it holds 8% of the idle lane-slots. So sorting by
material (section 5's first thought, Cycles' and pbrt-v4's answer) is not
where the time is here. Two throwaway counters on the same frame showed
where it is:

**The initial pass: path length.** How many surfaces each of the
259,200 traced pixels' paths walked (`path::walk_path`'s loop):

| Surfaces | 1 | 2 | 3 | 4 | 5 | 6 | 7 to 10 |
|---|---|---|---|---|---|---|---|
| Paths | 0.4% | 83.8% | 3.2% | 11.4% | 0.9% | 0.2% | 0.07% |

The mean is 2.30, and a warp's longest path 3.12 (over the 8,100
warps): with every lane in step through each surface, the loop's lanes
would be busy 73.5% of its surfaces, and the pass measures 67.3%. Most
of the initial pass's idle lanes are its sixth of paths (15.8%) that go
past the second surface, holding their warps.

**The spatial shift: the partners' outcomes.** Each pixel's three
partners (777,600 pairs), at `spatial_shift()`'s branches:

| Outcome | Pairs |
|---|---|
| shifted, reconnecting to a light | 71.3% |
| not paired (a store) | 22.6% |
| appended to the replay list | 5.4% |
| shifted, reconnecting to a surface | 0.5% |
| not held (a store) | 0.3% |

A thread walks its three partners in turn, so at each a warp's lanes
that only store wait for the lanes that shift: three lanes in ten idle
before the shift's own branches.

These are one frame's counts, as section 2's were 100 frames'.

**And what shipped since:** ReSTIR PT Enhanced (Lin et al. 2026,
section 6.2), NVIDIA's own, measured the same divergence in its
megakernel (active threads a warp 15.3 of 32, 19.9 after its changes)
and took four steps: branches made selects (6.2.1), the replay pairs
compacted and run one thread a pair (6.2.2), the forced reconnection to
NEE's lights (6.2.3), Russian roulette at the initial pass only (6.2.4).
This pass already has 6.2.2 to 6.2.4. It keeps the megakernel; no
wavefront.

### 8.2 The design: two changes, each its own

**1. The spatial shift over a list of the pairs that shift** (first: the
larger idle share, and the smaller change). ReSTIR PT Enhanced's 6.2.2,
taken from the replays to every pair that shifts.

- **Pairing** (a plain CUDA kernel: it traces nothing): one thread a
  traced pixel, its three partners as now; a pair not paired or not
  held stored as now; a pair that replays appended to the replay list
  as now; a pair that shifts appended to a new list of shift pairs (the
  same item, pixel and slot, `restir_pt::replay_item`), a warp's at once
  (`device/append.cuh`, GDSA.10).
- **Shift** (the OptiX launch it is now, over the list): one thread a
  pair, launched at the list's capacity (three a traced pixel), a thread
  past the count returning, as the replay launch is (Phase 2's
  measurement: cheaper than striding).
- **What it saves**, and what is not known: the pairs that only store
  stop holding lanes; the shift's own exits (a light not visible, the
  criteria) stay. Not estimated beyond that: the per-pass time says.
- **The picture: the same to the bit.** Each shift does the same
  arithmetic on the same inputs and writes its own slot; nothing
  accumulates. Guard: the 94 outputs, and the demo frame's bytes.
- **Costs:** one launch, a list of 4-byte items at three a traced
  pixel (12 MB at 3456 x 2234), its counter set back each frame with the
  replays'.

**2. The initial pass split after its second surface.** The walk
resumable, as section 4 made MNEE's: the first launch walks every path
through at most two surfaces; a path still going appends its state to a
list, and a second launch, at the list's capacity, walks those to their
ends. One split, not rounds: past the fourth surface 1.2% of paths are
left.

- **The state a path carries across:** the walk's (`path::Walk`, about
  60 bytes), the tree's gather (`restir_pt::TreeGather`: its reservoir
  of 64 bytes, the main path's vertex before and its reconnection
  vertex, the light seen; about 200 bytes), where its numbers are in
  their stream, and the surface it is at. 16% of the traced pixels:
  about 12 MB written, and read back, at 1920 x 1080.
- **What it should save:** by the histogram, the loop's warp-surfaces
  from 25,306 to about 19,850 (22% fewer), if the survivors' warps are
  as divergent among themselves as now (an estimate); about 0.4 ms of
  the pass's 2.38 if the loop is most of the pass. Modest, against a
  new resumable state and a second launch: the user's to weigh.
- **The picture:** each path's arithmetic the same. The survivors'
  MNEE requests are appended from the second launch, so MNEE's finish
  adds the solves' light in another order, float atomics' (as section
  7's rounds): the graphs without MNEE the same to the bit, the MNEE
  graphs' frame shown side by side.

**Considered, not taken:**

- **A wavefront with sorting by material** (Cycles', pbrt-v4's): the
  materials hold 8% of the idle lanes; it would move every path's whole
  state every surface for that.
- **Reconverging the lanes after the path loop** (its epilogue runs with
  14 lanes of 32): 0.6% of the pass's warp instructions.
- **Branches made selects** (ReSTIR PT Enhanced 6.2.1): line by line,
  after these two, if the per-line counters still show a branch worth it.

### 8.3 Separate, and the user's call: fast math

The counters' busiest lines in both passes are the IEEE operations:
`normalize`'s three divisions, square roots, sines and cosines (in the
initial pass 30% of the stall samples, in the shift 36%). The build
declines fast math so the GPU computes as the CPU does, which the GPU
tests compare against (`cmake/CudaModules.cmake`;
`2026-10-19-cuda-backend.md`, "Math mode"). Cycles (`--use_fast_math`,
`intern/cycles/device/cuda/device_impl.cpp`) and pbrt-v4 (its CMake)
build their GPU kernels with it. Taking it changes every graph's bits
and needs the GPU tests' comparisons decided again: not part of 8.2, and
measured only if the user wants the number.

## Sources

- Blender Cycles —
  https://github.com/blender/blender, intern/cycles/integrator/path_trace_work_gpu.cpp,
  intern/cycles/kernel/integrator/mnee.h, intern/cycles/kernel/types.h
- pbrt-v4 — https://github.com/mmp/pbrt-v4, src/pbrt/wavefront/surfscatter.cpp
- Falcor — https://github.com/NVIDIAGameWorks/Falcor, Source/RenderPasses/PathTracer
- RTXPT — https://github.com/NVIDIA-RTX/RTXPT
- Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013 —
  https://research.nvidia.com/sites/default/files/publications/laine2013hpg_paper.pdf
- Aila, Laine, "Understanding the Efficiency of Ray Traversal on GPUs",
  HPG 2009, and its 2012 addendum —
  https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02.pdf
- Hanika, Droske, Fascione, "Manifold Next Event Estimation", 2015 —
  https://jo.dreggn.org/home/2015_mnee.pdf
- Hong et al., 2025 (NVIDIA) —
  https://research.nvidia.com/labs/rtr/publication/hong2025partition/hong2025partition_paper.pdf
- OptiX 9 Programming Guide, shader execution reordering —
  https://raytracing-docs.nvidia.com/optix9/guide/pages/shader_execution_reordering.html
