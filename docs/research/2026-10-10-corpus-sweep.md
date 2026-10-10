# The whole codebase against the C++ guideline corpora

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864. The code at c0bc7d1 swept; the fixes merged at d1a5702.*

## Question

Before the reference render fixes the baseline, does every line of the
renderer hold to the two corpora the project's headers cite: the C++ Core
Guidelines and the C++ performance guidelines? Where it does not, which
departures are defects, which are deliberate, and which are not worth what
they cost?

## Method

Two sweeps, each of five reviewers, one per area: the core (`src/core`),
the Metal host code (`src/metal`, not shaders), the shaders (`*.metal`,
`*.metal.h`), the programs and build (`src/app`, `src/headless`, CMake,
tools), and the tests.

1. **Defects.** Each reviewer read its area and searched both corpora for the
   topics the code raised. Reports: `2026-10-10-corpus-sweep/report-s1-*.md`.
2. **Rule by rule.** Each reviewer listed every category of both corpora
   (Core Guidelines P, I, F, C, Enum, R, ES, Per, CP, E, Con, T, SF, SL, NL;
   performance memory, copy-move, cache-layout, lifetime, concurrency,
   codegen, gpu, gpu-dsa, cpu-dsa, telemetry) and checked its files against
   each rule, style and naming included, an ID on every finding, the rules
   walked and the rules found clean listed so coverage can be audited.
   Reports: `report-s2-*.md`. The first sweep had stopped at defects; this
   one was asked for the rules themselves.

Then five fixers, one per area, each in its own worktree, gave every row a
verdict, under one policy:

- **fix**: changed, with a test for every behavior change and every new
  error path; a test that could not fail was shown to fail by breaking the
  code it guards, then restored;
- **convention**: a deliberate project-wide departure, recorded once in
  `AGENTS.md` ("Guideline deviations") with its reason;
- **reject**: wrong, or not worth its cost, with the reason citing a rule.
  A speed claim that did not measure was rejected under Per.6 and GPU.10.

Their tables, one row a finding, with the commit that fixed it:
`2026-10-10-corpus-sweep/verdicts-{core,host,shaders,cross,tests}.md`.

## Results

| Area | Rows | Fix | Convention | Reject | Other |
|---|---|---|---|---|---|
| Core | 105 | 82 | 2 | 18 | 3 handed to other areas, all done |
| Metal host, programs, build | 126 | 110 | 3 | 8 | 5 fixed in part, rejected in part |
| Shaders | 64 | 46 | 2 | 11 | 5 handed to other areas, all done |
| Tests | 90 | 82 | 1 | 2 in part | 4 needing the source, all done |
| Across areas | 34 | rows of the lists above, done at the merge |

The defects that mattered most:

- **Undefined behavior:** a double cast to `int64` before its range check, in
  a flight's sample count and in a rhythm's flash number, reachable from
  numbers the scene reader accepted (ES.46); the shaders' rays ending at
  `INFINITY` while compiled with fast math, which assumes no infinities;
  glass and the clear coat dividing by zero at exactly grazing angles;
  uninitialized fields returned in two structs (ES.20).
- **Lifetime:** on an error mid-run, GPU memory freed while up to two frames
  still used it, because the object that waits for the GPU was destroyed
  last (C.31, R.1, GPU.9). A new test throws mid-run with frames in flight:
  without the fix the GPU faults, with it the run ends clean under Metal's
  validation layer.
- **Errors that never fired:** a scene or graph file whose read failed was
  parsed as empty ("no camera"), because the read's error was checked on the
  wrong stream (I.10); a camera at extreme coordinates passed its check and
  framed to NaN.
- **Tests that could not fail:** a resize check true for any image, means
  that dropped NaN samples silently, a conductor never tested with a colored
  f0, glass tested only head-on, and tolerances (doctest's `Approx` without
  `.scale(0)`) several times looser than they read.
- **The estimator:** Russian roulette scaled by radiance inside glass, so
  paths there ended twice as often as their energy warranted; fixed by the
  eta scale pbrt-v4 tracks (unbiased before and after). Each pixel's random
  numbers from a 32-bit key shared by thousands of pixel pairs a frame;
  now a 64-bit stream per pixel and frame (GDSA.3).

Also found by measuring rather than reading:

- Metal aborts the program on a texture side over 16384 instead of returning
  null; sizes are now refused before Metal is asked.
- The residency set holds a reference to what it contains.
- Without the barrier between frames, a frame's GPU time rises from 27.7 ms
  to 67–79 ms: the frames contend; the barrier stays.
- AddressSanitizer's runtime deadlocks before `main` on this macOS
  (`2026-10-10-sanitizers.md`), so the sanitizer preset is UBSan only.

## Measurements

Path tracer, `scenes/marbles.toml` (second look, c0bc7d1), 3456 x 2234,
median of frames 21 to 199, before and after alternated four times, each
run started after the GPU had been under 5% busy for six seconds:

| | Before (c0bc7d1) | After (d1a5702) |
|---|---|---|
| Frame | 27.53 ms (27.53–27.54) | 28.44 ms (28.44–28.48) |

The 0.9 ms is the roulette fix: paths inside glass now live as long as
their energy warrants (the shaders' table attributes +1.4 ms to it, partly
offset by the rest). No speed is claimed for any other fix; the one
proposed for speed alone, dropping the path pass's second image write
(GDSA.6), measured within noise and is kept for the 123 MB image it
removes (`2026-10-10-path-radiance-write.md`).

Loading the marbles' scene: 3 to 4 times faster, about 460 ms to 125 ms,
from placing each still shape once instead of checking its transform at
every distance query (CDSA.32, CACHE.4); profiled, the distance query was
92% of the load. Allocation, strings and TOML parsing were under 0.1% of
it, and `animate()` takes 43 µs a frame, which is the measured ground for
rejecting the allocator, copy and layout rows on those paths (Per.6).

GPU counters on the same frame after the fixes (Performance Limiters, as in
`2026-10-10-path-kernel-counters.md`, 5.67 s window): RT unit active 93.5%,
kernel occupancy 22.3%, the occupancy manager's target 25.7%, L1 eviction
rate 91.0%, L1 filled by ray tracing scratch 48.7% and registers 23.0%,
ALU 13.9%. The marbles' second look is the five spheres' picture, stronger:
ray traversal at a quarter occupancy, held down by per-thread state.

## Decision

Adopted: every fix, merged at d1a5702; the conventions in `AGENTS.md`;
`-Wconversion -Wsign-conversion -Wshadow -Wold-style-cast` on every
first-party target; Metal validation in ctest; a UBSan preset.

The time baseline the later estimators are measured against is the naive
path tracer at d1a5702 on the marbles: 28.44 ms a frame at 3456 x 2234,
with the counters above. The deferred change, keeping less state alive
across the shadow ray's query, stays deferred, so that baseline is the
textbook kernel.

Open: AddressSanitizer on a later macOS; a pinned clang-format before a
`.clang-format` is adopted.

Superseded: the marbles now open in the dark, and the time baseline is
measured again at a named time, by a method kept with it, in
2026-10-10-dark-opening.md ("The time baseline, again"): 29.90 ms at t =
120, every firefly awake. That method gives this note's code and scene
28.60 ms at t = 0, against the 28.44 ms above.
