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

## 8. The tracing passes as a wavefront, as Cycles runs its integrator

Status: design, 2026-10-21, for the user's review before any header or
code. It replaces the first draft of this section (2958b6c), which split
the initial pass after its second surface because one frame of the
marbles ended most paths there: a parameter fitted to one scene, not a
design. The rule here: take the production design whole, and set each of
its parameters from the 3090's properties and the programs' own (their
registers, spills and state bytes), never from a scene's statistics.
Scenes confirm a design; they do not choose its numbers.

### 8.1 What the hardware does with the megakernels now

Nsight Compute's source counters on the demo's frame 7000 at 1920 x 1080
(a throwaway build with line information, 160 registers): the initial
pass 2.38 ms with its lanes 67.3% busy, the spatial shift 1.20 ms at
48.0%. In both, every line inside a path's loop or a partner's loop runs
with the lanes of whichever paths are still there; lanes that leave a
loop wait, and after the loop they do not run together again (the
initial pass's epilogue, its reservoir written, runs with 14 lanes of
32). That is the megakernel's shape on a GPU without shader execution
reordering, whatever the scene: a warp runs as long as its longest
path, through every branch any of its paths takes (Laine, Karras and Aila
2013; section 3). Two throwaway counters on that frame show it there (the
paths' lengths: a warp's longest 3.12 surfaces against a mean of 2.30;
the shift's partners: 71% shift, 29% only store or append) and nothing
more: another scene moves both.

### 8.2 What production does

- **Cycles' GPU integrator** (`intern/cycles/integrator/
  path_trace_work_gpu.cpp`, `kernel/integrator/`):
  - a path's state in arrays, one a member (structure of arrays), so a
    kernel reads only the members it uses, coalesced;
  - the work as kernels each with one job: from the camera, intersect
    closest, shade the background, shade a surface, intersect a shadow
    ray, shade a shadow, intersect MNEE's seed;
  - each kernel launched over the paths queued for it, their indices
    gathered first (`integrator_queued_paths_array`): compaction before
    every launch, of every kernel, at no chosen point;
  - the surface shading launched over its paths sorted by shader (a
    counting sort, `kernel_uses_sorting`), in partitions of 65,536
    states for the locality of their state's reads
    (`num_sort_partitions`);
  - shadow rays in their own states and kernels;
  - as many states as `max(SMs x threads an SM, 65,536) x 16` (the CUDA
    queue's `concurrent_states_params`), against device memory;
  - no explicit register limit (OptiX's own choice), and `--use_fast_math`
    (`device/cuda/device_impl.cpp`).
- **pbrt-v4's GPU integrator**: the same wavefront, a queue a kernel and
  a kernel a material (`src/pbrt/wavefront/`), `--use_fast_math`.
- **ReSTIR PT Enhanced** (Lin et al. 2026, NVIDIA's ReSTIR PT, section
  6.2): the shifts run one thread a pixel and neighbor pair, compacted
  (6.2.2); branches made selects (6.2.1); the forced light reconnection
  (6.2.3) and roulette at the initial pass only (6.2.4), both already
  here.

### 8.3 The design

**1. The initial pass as a wavefront** (CUDA: this note is the 3090's;
Metal's design is its own).

- **State.** A path's state in arrays, a member each (Cycles'): the
  walk's (`path::Walk`: throughput, radiance, the ray, the medium, MIS's
  density, where it left from), the tree's gather
  (`restir_pt::TreeGather`: the reservoir, the main path's vertex before
  and its reconnection vertex, the light seen, MNEE's chain), where its
  numbers are in their stream, its surface number, and the hit its
  intersect left. One state a traced pixel: Cycles' count for this GPU,
  `max(84 x 1,536, 65,536) x 16` = 2,064,384, holds every traced pixel of
  the largest size in play (965,088 at 3456 x 2234), so the frame's paths
  are all in flight at once and none waits for a state. Made once at the
  largest size (MEM.9).
- **Kernels**, each launched over a compacted list of the paths queued
  for it, appended a warp's at once by the kernel before
  (`device/append.cuh`, GDSA.10):
  - *camera* (CUDA): the primary ray, the gather started; every path to
    intersect.
  - *intersect closest* (OptiX, a ray generation program that traces
    and nothing else): the hit into the state; a miss to *background*,
    a hit to *surface*, with the hit's material as its sort key.
  - *background* (CUDA): the sky's light offered to the gather (step 2);
    the path ends.
  - *surface* (CUDA, sorted by material): emission offered (step 3), the
    BSDF resolved (step 4), the light sample drawn and weighed (step 5)
    and, if it gives light, a shadow ray queued with its candidate
    whole; MNEE's seed queued where its chain is; the BSDF sampled
    (step 6), roulette (step 7), the next ray queued to intersect (step
    8) or the path ended.
  - *intersect shadow* (OptiX): visibility; *shade shadow* (CUDA): the
    candidate offered to the gather if the light is visible.
  - *MNEE's seed* (OptiX): the seed's segment traversed and the request
    appended, as now.
  - *finish* (CUDA): the reservoir, the primary hit, last frame's
    position and the light seen written, as the pass's epilogue now.
- **The schedule.** Cycles picks the next kernel on the host from the
  queues' counts, read back each iteration: for an offline render. A
  frame cannot stop for a read-back each step, and a ReSTIR PT path has
  at most `max_surfaces` surfaces (14 by contract, 9 in the shipped
  graphs), so the schedule is fixed: for each surface, intersect
  closest, background, surface, intersect shadow, shade shadow, MNEE's
  seed, each launched at its list's capacity with the threads past its
  count returning (Phase 2's measurement: cheaper than striding). Every
  list is the previous step's compacted output, so each launch's warps
  hold only paths with that work.
- **The order a path offers its candidates is the megakernel's.** At a
  surface the megakernel offers emission, then the light sample, then
  scatters; here the shadow's candidate is offered by *shade shadow*,
  after *surface* has scattered, so the shadow state carries everything
  the offer reads at that moment (the integrand, the stored path, the
  roulette survival so far) and *shade shadow* runs before the next
  surface's offers. Each path's reservoir then sees the same candidates
  with the same numbers in the same order: its pick the same to the bit.
- **Sorting** (Cycles' counting sort): *surface*'s list ordered by
  material, in partitions of 65,536 states, Cycles' number, for the
  locality of the states' reads (an inference, to be measured: at the
  hundred-odd bytes a surface reads of a state, a partition's reads are
  about the 3090's 6 MB L2). The key is the scene's material, as Cycles'
  is its shader.
- **Registers**: each kernel compiled at the top of the occupancy step
  that holds its registers with no direct spills, read from OptiX's
  compile report for the OptiX programs and nvcc's for the kernels
  (the GA102's steps: an SM's 65,536 registers over 32 lanes a warp,
  allocated in eights). The rule is the programs' own; section 7's one
  limit of 160 for every program becomes the rule applied to each.

**2. Both shifts over lists of pairs** (ReSTIR PT Enhanced's 6.2.2, on
every pair that shifts, not only the replays). The temporal and the
spatial shift alike:

- *pairing* (CUDA: it traces nothing): one thread a traced pixel and
  its partners as now; the pairs that only store, stored; the pairs that
  replay appended to the replay list as now; the pairs that shift
  appended to a shift list.
- *shift* (OptiX): one thread a pair of the shift list, launched at its
  capacity; then the replays as now.
- Each pair's shift is the same arithmetic into its own slot: the same
  to the bit.

**3. MNEE's rounds' length from the program.** Cycles walks MNEE whole
in its own kernel; on Ampere the rounds stay (section 4), and their 8
iterations, chosen from the marbles' walks, become the program's: the
fewest iterations whose arithmetic outlasts carrying a walk's 96-byte
state out and back in (the round kernel's cycles an iteration from
Nsight Compute, against the state's bytes at the GPU's bandwidth).

**4. Fast math** (Cycles', pbrt-v4's). The counters' busiest lines in
both passes are IEEE division, square root, sine and cosine (30% of the
initial pass's stall samples, 36% of the shift's), which
`--use_fast_math` makes hardware instructions. The build declines it so
the GPU computes as the CPU does, which the GPU tests compare against
(`cmake/CudaModules.cmake`; `2026-10-19-cuda-backend.md`, "Math mode").
Taking it changes every graph's bits and the GPU tests' comparisons
need tolerances: listed here with the production design, the user's
call to take.

### 8.4 The picture and the guards

- Changes 1 and 2 keep each path's and each shift's arithmetic and each
  reservoir's order: the graphs without MNEE the same to the bit (the 94
  outputs). MNEE's requests are appended from other launches, so the
  finish's float atomics add in another order: the MNEE graphs' frame
  side by side, as section 7's rounds.
- Change 3 changes when a walk is regathered, not its arithmetic: the
  same.
- Change 4 changes every bit: the demo's frame side by side.
- Tests (invariants, not settings): the state's layout (host and device
  agree, as `MneeWalk`'s); the wavefront's paths equal to the megakernel's
  on built inputs (the same reservoir to the bit), the megakernel
  removed once that holds (a decision removes the alternative).

### 8.5 Phases, and the way back

1. The shifts' pair lists (change 2): the smallest step, its own commit.
2. The wavefront's state and kernels beside the megakernel, the test of
   their equality; then the pass switched and the megakernel deleted.
3. Sorting, then each kernel's register limit.
4. MNEE's round length from its program.
5. Fast math, if the user takes it.

Each phase measured on the demo (fps at both sizes), the per-pass times
for the passes it changes, and gated. A phase that loses is reverted by
its commit; the megakernel is in git until phase 2's switch holds.

### 8.6 Results

**Phase 1, the shifts over lists of pairs** (2026-10-21, change 2 as
designed). The demo, release, the GPU otherwise idle, headless
`--times` over frames 7000 to 7599; before is the work branch at
d6f8dd6 (the 160-register limit), built and run the same hour,
alternated, twice each:

| Size | Before | After |
|---|---|---|
| 1920 x 1080 | 117.0 fps (8.55 ms) | **129.7 fps (7.71 ms)**, +10.9% |
| 3456 x 2234 | 34.8 fps (28.77 ms) | **37.5 fps (26.68 ms)**, +7.8% |

The picture: the marbles' outputs the same to the bit; seven of the
brass sphere's ReSTIR PT frames differ, in at most 0.15% of their pixels
and each by one level of 255, the same twice over (no race). The shift
is the same arithmetic, now compiled in a program of its own, where the
compiler fuses its multiplies and adds differently (an inference, not
shown). Shown to the user side by side.

### 8.7 What the first draft got wrong, recorded

- *The initial pass split after its second surface*: withdrawn, fitted
  to the marbles' path lengths; the wavefront compacts after every
  surface by construction.
- *"Sorting by material is not where the time is"*: true of one frame of
  a scene with few materials, not of the design; sorting is Cycles' and
  is taken.
- *Section 7's one register limit of 160*, chosen by one scene's frame
  times: kept until phase 3 replaces it by the per-program rule.

## Sources

- Blender Cycles —
  https://github.com/blender/blender, intern/cycles/integrator/path_trace_work_gpu.cpp,
  intern/cycles/kernel/integrator/mnee.h, intern/cycles/kernel/types.h,
  intern/cycles/kernel/integrator/state_template.h, intern/cycles/device/queue.cpp,
  intern/cycles/device/cuda/queue.cpp, intern/cycles/device/cuda/device_impl.cpp,
  intern/cycles/device/optix/device_impl.cpp
- Lin et al., "ReSTIR PT Enhanced", 2026 —
  https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/
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
