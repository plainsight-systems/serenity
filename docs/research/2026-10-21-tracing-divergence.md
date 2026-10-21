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
not one loop, and how much of their time it costs is not measured:
reading each program's SIMT efficiency (the lanes busy per instruction)
and registers needs Nsight Compute, which needs the GPU's performance
counters, a root setting on the 3090 (NVIDIA's ERR_NVGPUCTRPERM). With
those numbers, their design follows (a wavefront split with sorting by
material, Cycles' and pbrt-v4's, is the production answer, and the
larger change).

## 6. Steps

1. MNEE's walk in rounds (section 4). Measured: the demo's frame,
   before and after, at both sizes, in frames a second; the solve's
   three steps' GPU times. Gates; the 94 outputs against
   `/data/serenity/identity/phase1.sha` for the graphs without MNEE.
2. With the user's leave for the counters: the initial pass's and the
   spatial shift's SIMT efficiency and registers, then their design.

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
