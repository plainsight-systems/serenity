# What the split of the path loop cost the path pass

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864, `-std=metal4.0 -fmetal-math-mode=fast -O2`. Release builds of
main (4b61224), ReSTIR DI's Phase A (f86e7ae) and variants of Phase A's
`src/metal/integrator/path.metal.h`.*

## Question

Phase A split `path.metal.h`'s `radiance()` into `camera_segment()` and
`continue_path()`, two loops built from shared step functions, `arrive()`
(steps 1 to 4) and `scatter()` (6 to 8), so that ReSTIR's passes share the
naive path tracer's light transport (ES.3). The path pass called both in
turn. Its images stayed the same to the bit, but its frame time rose by
about 19% at every time measured. ReSTIR is additive: the naive path tracer
keeps its images and its time. What in the split costs the time, and what
shape of shared code costs none?

## Method

- **Timing.** The path tracer (`graphs/path.toml`) on `scenes/marbles.toml`
  at 3456 x 2234. Each run renders 201 frames with two in flight; the
  figure is the median GPU time of frames 21 to 199. Before each run, no
  serenity process is running and `ioreg`'s Device Utilization reads at
  most 5% for six consecutive seconds. Builds are alternated within each
  round. The harness is a temporary doctest case, never committed.
- **Bisection.** Variants of `path.metal.h` were built inside f86e7ae's
  tree, the rest of the tree unchanged. One variant was main's `radiance()`
  text pasted back in.
- **Static evidence.** `path.metal` was compiled to AIR and to LLVM IR with
  the build's flags. The `path_trace` pipeline was then compiled on the
  device and serialized through a binary archive. From the archive's GPU
  slice (`applegpu_g15s`), the size of the `__compute` section (the kernel's
  machine code) was read and its bytes compared. The pipeline's
  `maxTotalThreadsPerThreadgroup` was 1024 for every variant, so it did not
  separate them. No disassembler for the GPU's code is available.
- **Counters.** The Performance Limiters recording
  (2026-10-10-path-kernel-counters.md) saved once in twelve attempts:
  `xctrace record` segfaulted while saving the other eleven, for every
  binary, with 8, 11 and 14 s windows. Only main's run has counters.
- **Images.** Each build's `serenity-headless` rendered
  `--format pfm --size 160x96 --frames 3 --samples 4 --write all` for each of
  the four scenes, at `--time 0`, `8` and `120` and moving
  (`--step 0.5 --first 3`): 48 PFMs. The outputs were compared byte for byte
  with main's build, which was rebuilt from 4b61224 for this check.

## Results

### The regression, and where it is not

| Build | t = 120 (ms) | `__compute` (bytes) |
|---|---|---|
| main 4b61224 | 29.64 | 58,768 |
| f86e7ae (Phase A) | 35.21 | 94,320 |
| E0: f86e7ae with main's `radiance()` pasted back | 29.67 (main 29.66 beside it) | 58,768, the same bytes as main's |

E0's IR for `path_trace` matches main's, apart from the file path, and its
GPU code is byte-identical. So nothing outside `path.metal.h`'s loop costs
the path pass any time: not the bindings, the renderer's changes, or the
headers Phase A added to.

### The cause

Facts, from f86e7ae's IR and GPU code:

- **The transport is compiled twice into the path kernel.**
  - `arrive()` and `scatter()` are inlined into two loops, `camera_segment()`'s
    and `continue_path()`'s.
  - Each loop carries its own trace, surface interaction, BSDF resolution
    with texture evaluation, BSDF sampling and roulette.
  - The kernel's machine code grows from 58,768 to 94,320 bytes (+60%).
- **`continue_path()` carries a whole surface across every ray query.** Its
  loop variable `Shading here` (a 64-byte SurfaceInteraction, the Bsdf, wo
  and the medium) is assigned from the segment before the loop. After the
  first surface, it is replaced only after the next trace. So its old value
  is live across that trace. The IR's phi count rises from 123 (main) to 439.

The two parts separated, by experiment E5:

- E5 keeps two loops called in turn, as Phase A does, but carries no Shading
  across a trace. Steps 6 to 8 at the segment's surface come first, then a
  plain loop.
- E5 takes 32.84 ms (main 29.64 beside it), with 113,680 bytes of machine
  code: `next_event()` and `scatter()` are now inlined twice as well.
- So of Phase A's +5.6 ms, about 3.2 ms comes from compiling the walk as
  two loops. About 2.4 ms comes from the Shading carried across the queries.
- Inference: E5 differs from Phase A in more than the carried Shading. Its
  code is larger still, so the division is approximate.

Inference, from 2026-10-10-path-kernel-counters.md: the kernel runs at 22%
occupancy, limited by per-thread state that evicts L1. These are the two
ways the kernel would hold more per thread: a second copy of the loop's
code, and about 120 more bytes of state live across each query. Counters
on Phase A's kernel would show which limiter moved, but they could not be
recorded (Method).

Main's counters, recorded once at t = 120 (29.70 ms run):

| Counter | Mean |
|---|---|
| RT Unit Active | 93.8% |
| Kernel Occupancy | 22.0% |
| Occupancy Manager Target | 25.4% |
| L1 Eviction Rate | 91.0% |
| L1 RT Scratch Residency | 49.5% |
| L1 Register Residency | 23.0% |
| L1 Stack Residency | 0.0% |
| ALU Utilization | 13.6% |
| Compute SIMD Groups Inflight | 21.1 |

### One loop: what each shape cost

Each variant is one loop for the path pass, built from the same steps. They
are listed in the order tried, at t = 120, each figure the variant's median
beside main's in the same round. "Bits" compares the marbles at t = 8, t =
120 and moving with main's build; the marbles are the one scene where any
variant differed.

| Variant | Runs (ms; main beside) | Δ vs main | `__compute` | Bits |
|---|---|---|---|---|
| E1: `radiance()` one loop over Phase A's `arrive()`/`scatter()` | 29.94 (29.69), 29.94 (29.67), 29.93 (29.65), 29.95 (29.66), 29.96 (29.66), 29.94 (29.66) | +0.9% | 58,512 | same |
| E3: E1 with the steps `always_inline` | 30.19 (29.66) | +1.8% | 58,624 | not checked |
| E4: E1 with `arrive()` split at step 2 (no zeroed Shading on escape) | 28.99 (29.66), 29.03 (29.65), 29.04 (29.65) | −2.1% | 58,704 | 9 of 9 differ |
| F1/F2: one loop as a template (`Stop`) over a `Walk&`, steps as functions | 29.50 (29.65), 29.50 and 29.50 (29.65) | −0.5% | 58,928 | 9 of 9 differ |
| F4: F2 with steps 1 and 2 in the loop, all `always_inline` | 29.87 (29.68), 29.86 (29.66), 29.88 (29.65) | +0.7% | 59,088 | same |
| X6: F4 with steps 6 to 8 in the loop too | 29.92 (29.65), 29.89 (29.64) | +0.8% | 59,120 | not checked |
| X7: F4 with steps 1 to 5 in the loop, `scatter()` a function | 29.88 (29.66), 29.85 (29.68) | +0.7% | 59,088 | not checked |
| X8: every step in the loop, the state in a `Walk&` | 29.90 (29.64), 29.89 (29.66) | +0.8% | 59,120 | not checked |
| X9: the state in locals, `Shading here` declared before the loop | (IR only) | | | |
| X10: the state in locals, steps 6 to 8 a forced-inline lambda | 29.39 (29.65), 29.43 (29.66) | −0.9% | 58,784 | same |

Facts:

- X8's IR for the loop differs from main's only in the order of the loop
  header's phis (fields of a `Walk&` rather than locals) and in one
  `and i8 1` on the emission flag. It is still +0.8%.
- X9 kept `Shading here` loop-carried even in the camera instantiation, with
  zero-initialized phis at the loop header. It was dropped before timing.
- E4 and F2 differ from main by 1 to 3 ulps (relative ≤ 4.1e-7) in 597 to
  680 of 46,080 values of each differing marbles image. The other scenes
  were the same.

Inferences:

- Once the loop exists once, what remains is ±2% of code-generation
  variance between semantically identical shapes: the order in which the
  compiler keeps the state, inlining at the AIR level, and select against
  branch at step 7. The GPU compiler's register allocation is not visible
  here, so the mechanism is not identified.
- Under fast math, the compiler may reassociate. A shape that changes the
  IR can change the rounding, even with the same operations in the source.
  The images are the arbiter of "the same to the bit", not the source.

### The fix

`walk_path<Start, Stop>()` is one loop, and its start and stop are template
parameters:

| Caller | Instantiation | What it walks |
|---|---|---|
| `radiance()` | `<camera, at_end>` | the path pass's loop |
| `camera_segment()` | `<camera, at_first_aimed>` | ReSTIR's initial pass, stopping after step 4 at the first surface that aims at lights |
| `continue_path()` | `<after_step_5, at_end>` | ReSTIR's shade: steps 6 to 8 at the given surface, then the loop |

- The state lives in locals, as main's loop kept it.
- Steps 1 to 5 sit in the loop's body. Steps 6 to 8 are a lambda over the
  locals (F.50, F.52), because the continuation also runs them once before
  the loop.
- The function and both lambdas are forced inline (GEN.6, a deliberate
  override).
- `FirstLight` is gone. The path pass no longer runs through
  `continue_path()`, so its only remaining caller is ReSTIR's shade, which
  always leaves step 5 to itself.
- `continue_path()` no longer carries the surface across later queries: the
  surface's steps 6 to 8 run before the loop.
- The final build's `path_trace` GPU code is byte-identical to X10's: 58,784
  bytes, 16 more than main's.

### Final measurements

Three rounds, builds alternated in each round:

| Graph, time | Round 1 | Round 2 | Round 3 | Δ |
|---|---|---|---|---|
| path t = 120, main / fix | 29.66 / 29.40 | 29.66 / 29.38 | 29.65 / 29.42 | −0.9% |
| path t = 8, main / fix | 31.58 / 31.32 | 31.55 / 31.30 | 31.56 / 31.30 | −0.8% |
| path t = 0.5, main / fix | 21.93 / 21.92 | 21.92 / 21.93 | 21.89 / 21.95 | +0.1%, within the spread |
| restir_ris t = 120, Phase A / fix | 39.84 / 37.15 | 39.85 / 37.15 | 39.85 / 37.13 | −6.8% |

Images:

- All 48 of the path pass's PFMs are byte-identical to main's.
- ReSTIR RIS's 48 PFMs are not byte-identical to Phase A's under fast math.
  44 of 48 differ. In a first frame, the differences are below 1e-5
  relative. By the third frame, a few values differ by up to 6%, where a
  rounding difference flipped a discrete choice.
- Built with `-fmetal-math-mode=safe`, the fix and Phase A render all 48
  RIS PFMs and all 48 path PFMs identically. ReSTIR's computation is
  unchanged; only fast math's rounding moved.

Tests: native-release, native-debug and native-sanitize build with no
warnings, and ctest passes on each (6 of 6, the GPU tests under Metal
validation). `make check` passes.

## Decision

Adopted: one loop, `walk_path<Start, Stop>()`, with the state in locals.

- The path pass is no slower than main: 0.9% faster at t = 120 and t = 8,
  the same at t = 0.5.
- Its images are byte-identical to main's.
- ReSTIR's three uses share every line of the transport (ES.3).

Rejected:

- Two loops called in turn for the path pass (Phase A, E5).
- Shapes that change the path pass's rounding (E4, F1/F2).
- Shapes that keep the bits but are 0.7 to 0.9% slower (E1, F4, X6, X7, X8).
- A `Shading` declared before the loop (X9).

What the 0.9% gain comes from is not established (Results, inferences). It
is reported here, not claimed by the header. Open: counters for Phase A and
the fix, when `xctrace` saves them.
