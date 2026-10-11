# ReSTIR DI Phase A against the two guideline corpora, rule by rule

*2026-10-10. The implementation of ReSTIR DI without reuse on branch
`restir-di-phase-a`, from cef7923: the path tracer's split, contracts 4, 5
and 14, the light points, the target, the reservoir's candidates and
finish, `restir_initial` and `restir_shade`, the core's pass kinds,
settings and rules, `ReuseImages` for a graph without merges,
`graphs/restir_ris.toml` and the Makefile's `GRAPH`. Both corpora answered:
the C++ Core Guidelines through its MCP server, the performance guidelines
through its HTTP endpoint (the `cpp-perf-guidelines` MCP client was not
connected; the endpoint was). Every category of each was listed fresh and
walked over the diff; every finding has an ID and a verdict, and every
valid one is fixed in the commit.*

## Scope

New: `src/metal/passes/restir_initial/restir_initial.{cpp,metal}`,
`src/metal/passes/restir_shade/restir_shade.{cpp,metal}`,
`graphs/restir_ris.toml`, `tests/restir_schedule_test.cpp`,
`tests/restir_graph_file_test.cpp`, `tests/gpu/restir_test.cpp`,
`tests/gpu/kernels/restir_probe.metal`.

Implemented in, or changed: `core/frame/schedule.{h,cpp}`,
`graph_file.cpp`; `core/contracts/{light_sample,bsdf,surface_record}.h`;
`core/passes/restir_di.h`; `metal/math/warp.metal.h`,
`metal/sampler/sampler.metal.h`, `metal/light_selection/
uniform_light.metal.h`, `metal/lights/{sphere_light,emitter}.metal.h`,
`metal/materials/bsdf.metal.h`, `metal/integrator/{path,target}.metal.h`,
`metal/reuse/{reservoir,unbiased}.metal.h`, `metal/passes/restir_di/
restir_di.metal.h`, the two pass headers, `metal/frame/{frame_images,
renderer}.{h,cpp}`; `CMakeLists.txt`, `tests/**/CMakeLists.txt`,
`tests/gpu/kernels/probes.h`, `Makefile`.

## The path pass, unchanged

The split of `path.metal.h` into `camera_segment()` and `continue_path()`
(over `arrive()`, steps 1 to 4, and `scatter()`, 6 to 8) leaves the path
pass's accumulated image bit-identical. Checked by rendering
`graphs/path.toml` with `serenity-headless --format pfm` (release build) at
160 x 96, 3 frames of 4 samples, for each of the four scenes in `scenes/`,
at `--time 0`, `8` and `120` and moving (`--step 0.5 --first 3`): 48 PFMs,
before (a build of 4b61224, whose path pass the design commits changed only
by comments and declarations: `git diff --stat 4b61224 cef7923` over the
path pass's shaders shows insertions only) and after; `diff -r` finds none
differ. Each run was

    serenity-headless --graph graphs/path.toml --scene scenes/<scene>.toml \
        --out <dir> --size 160x96 --frames 3 --samples 4 --write all \
        --format pfm (--time 0 | --time 8 | --time 120 | --step 0.5 --first 3)

with the two builds' binaries, and `diff -r` over the two output trees.
The images are not empty: the marbles at t = 120 have a mean of 0.177 and
44274 of 46080 values non-zero. The frame time against the 29.90 ms
baseline is left to be measured on an idle GPU.

## Findings and verdicts

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| A1 | F.3 | core/frame/graph_file.cpp | `read_restir_di` did four things in 110 lines: decide what is read, refuse what is not, build the baseline, read and check each value | fixed: `RestirReads` (with `reads()`), `restir_reads()`, `refuse_unread()`, `restir_baseline()`; the reader composes them |
| A2 | I.24 | metal/integrator/target.metal.h | `candidate_weight(surface, light, float selection, float solid_angle_density)`: two adjacent floats, multiplied, either order compiles | fixed: one `solid_angle_density`, the selection probability folded in by the caller |
| A3 | ES.45 | metal/materials/bsdf.metal.h, warp.metal.h, restir_di.metal.h | the packed words' shifts and masks (16, 0xffff, 0xff, 0x7fff, 1) as literals | fixed: `packed_half_mask`, `packed_high_shift`, `packed_kind_shift`, `packed_kind_mask` beside `PackedBsdf` in contract 2; `half_magnitude`, `half_least`; `unit_code_mask`, `unit_y_shift`; `path_surface_mask` beside the path state's other fields in contract 5 |
| A4 | ES.101, ES.102 | restir_di.metal.h, `takes_test` | the test's signed 8-bit offsets sign-extended by shifting a signed int right, whose result for a negative value the language leaves to the implementation | fixed: the bits taken apart unsigned and reinterpreted as `char` by `as_type` |
| A5 | F.21 | metal/reuse/reservoir.metal.h | the candidates' loop had to learn whether `stream_candidate` kept its candidate by comparing identities before and after | fixed: `stream_candidate` returns whether it kept it (header changed) |
| A6 | ES.3 | light_selection/uniform_light.metal.h | `select_light_index` repeats `select_light`'s formula | kept, stated: writing `select_light` over it would change the path tracer's code for nothing it needs, against the additive rule; three lines |
| A7 | GPU.3 | restir_di.metal.h, `candidates` | the survivor's whole `LightReceived` (12 words) is kept beside the stream for visibility reuse's ray, which reads five | kept, stated: the fields the ray does not read are dead and the compiler drops them; the counters, not this note, would show otherwise (GPU.10) |
| A8 | I.13 | restir_di.metal.h, `first_surface` | last frame's transforms passed as a pointer | convention: MSL has no span in the constant address space; every scene array here is a pointer indexed by a primitive the structure returned (AGENTS.md's MSL deviations, ES.27's family) |
| A9 | F.9 | frame_images.cpp | `prepare(inputs, size)` and `addresses(index)` do not read `inputs` or `index` until the temporal merge | holds: unnamed, the comment naming what reads them later |
| A10 | P.5, E.2 | metal/frame/renderer.cpp | restir_temporal and restir_spatial reach a switch case | holds: refused by `MetalError` naming the pass, at construction, never skipped; tested |
| A11 | I.5 | metal/frame/frame_images.h | `ReuseImages` asked for the temporal arrays or the working array | holds: refused by `MetalError` at construction until the merges; tested |
| A12 | F.17 | path.metal.h, `arrive`/`scatter` | the walk passed as `thread Walk&` | holds: an in-out parameter |
| A13 | F.20 | graph_file.cpp, `restir_reads` | fills two settings through a reference beside its result | kept: the settings are in-out there, zero in and the two deciding values out, which the reader goes on filling (F.17) |
| A14 | GDSA.18 | restir_di.metal.h, `shade` | the record unpacked twice in the shade, once for the direct light, once for the continuation | holds, as the design chose: decoded in the load path, not held across the query (restir_di.metal.h) |
| A15 | GPU.8 | renderer.cpp | barriers | holds: the queue barrier before the initial pass (it writes the arrays the frame before's shade reads), one dispatch barrier before the shade (it reads what the initial pass wrote), one before the presenting pass, as before |
| A16 | GPU.9, MEM.9 | frame_images.cpp | the arrays | holds: made in `prepare()` at a size, before the submission, remade only when the size changes, drained first |

## Every category, rule by rule

Notation: **holds** (applies, and the diff keeps it), a finding number,
**convention** (AGENTS.md "Guideline deviations"), or **n/a** with why.

### C++ Core Guidelines (listed fresh: In, P, I, F, C, Enum, R, ES, Per, CP, E, Con, T, CPL, SF, SL, A, NR, NL, GSL, Tools, FAQ, Discussion)

**In.0** n/a (introduction).

**P** — P.1 holds (each step carried by number). P.2 holds for host
code; MSL by decision. P.3 holds (names: `arrive`, `scatter`,
`candidate_weight`, `takes_test`). P.4 holds (layouts asserted). P.5
holds (new kinds fail every switch's build; sizes asserted). P.6 holds
(settings checked at read, refused by name). P.7 holds (refusals at
construction and `prepare()`, before the submission). P.8 holds
(`ReuseImages`' arrays `Resident`). P.9 holds (no evaluation repeated: the
survivor's answer kept, A7). P.10 holds. P.11 holds (the visibility test
in one function, `restir::visible`). P.12 holds (tests under Metal's
validation; mutation runs, the report's). P.13 n/a.

**I** — I.1 holds. I.2 holds. I.3 n/a. I.4 holds (`Temporal`, `Working`,
`ReservoirOutput`, `FirstLight`). I.5 holds (preconditions stated:
`candidates`, `continue_path`, `select_light_index`). I.6, I.8 n/a in MSL;
host preconditions thrown. I.7 holds (the softmax and convergence tests
hold the postconditions). I.9 n/a (no templates in interfaces but the
merges', declared). I.10 holds. I.11 holds. I.12 n/a. I.13 A8. I.22
holds. I.23 holds (at most six parameters, `shade`'s). I.24 A2. I.25-I.27
n/a. I.30 holds (`leave()` used from a two-field surface, stated).

**F** — F.1 holds. F.2 holds. F.3 A1; `candidates`, `shade`,
`direct_from` under 40 lines. F.4 n/a (shader functions; the host's do
I/O). F.5 holds (shader functions inline). F.6 holds (`addresses()`
noexcept). F.7 holds. F.8 holds (warp, packing, target pure). F.9 A9.
F.10 holds (`scattered`, `area_factor` named once). F.11 holds. F.15,
F.16 holds (small structs by value in MSL). F.17 A12, A13. F.18, F.19
n/a. F.20 holds but A13. F.21 A5; `Arrival`, `Direct`, `Shaded`. F.22-F.27
n/a. F.42-F.49 hold (no references returned to locals). F.50-F.54 hold
(lambdas capture by reference locally). F.55 n/a. F.56 holds. F.60 n/a.

**C** — C.1 holds (`Walk`, `RestirReads`). C.2 holds (structs of
independent members; `ReuseImages` a class with an invariant). C.4, C.5
hold. C.8, C.9 hold. C.20, C.21 hold (`ReuseImages` deletes copy and move,
as `FrameImages`). C.31 holds. C.35-C.37 hold. C.41, C.42 hold (refusals
throw from the constructor). C.45-C.49 hold. C.90 convention: the reuse
constants copied into their ring by `memcpy`, trivially copyable asserted
(the codebase's FrameArray rule). Others n/a (no hierarchies, operators,
unions, containers).

**Enum** — Enum.3 holds (enum classes). Enum.7, Enum.8 convention (shared
with Metal). Others hold or n/a.

**R** — R.1 holds (`NS::SharedPtr`, `Resident`, `std::unique_ptr` for the
renderer's members). R.3 holds (raw pointers non-owning). R.11 holds (no
naked new). R.20-R.37 hold or n/a.

**ES** — ES.3 A6. ES.5, ES.6 hold. ES.10, ES.11 hold. ES.12 holds
(-Wshadow; `kinds` renamed in the reader). ES.20 holds (every struct
`{}`-initialized). ES.21, ES.22 hold. ES.23 convention in MSL. ES.25 holds.
ES.26 holds. ES.27 convention (`SoftmaxQuery::scores`). ES.28 holds. ES.30-
ES.34 n/a. ES.40, ES.41 hold. ES.42 holds. ES.43, ES.44 hold. ES.45 A3.
ES.46 holds (counts narrowed after a range check; floats by
`finite_float`; the 16-bit quantizations stated in the contracts). ES.47
holds (`nullptr`). ES.48, ES.49 hold (named casts; MSL conversions by
convention). ES.50 n/a. ES.55 holds. ES.56 n/a. ES.60-ES.65 hold. ES.70
holds (switches on kinds). ES.71-ES.74 hold. ES.75, ES.76 n/a. ES.77
holds (the walk's `break`s are the loop's ends, as the path tracer's
were). ES.78 holds (no fallthrough but grouped labels). ES.79 holds (no
`default`). ES.84-ES.87 hold. ES.100 convention (`uint` in shaders). ES.101,
ES.102 A4. ES.103, ES.104 hold (pixel counts in `size_t`). ES.105 holds
(divisions guarded: density, piSum, distance). ES.106, ES.107 convention.

**Per** — Per.1-Per.4 hold (no optimization without reason: the cancelled
weight and the kept survivor each stated). Per.7 holds (settings data).
Per.10, Per.11 hold (`constexpr` tables, settings checked once). Per.19
holds (flat arrays). Others n/a.

**CP** — n/a (no CPU threads; the GPU's ordering is the barriers').

**E** — E.1, E.2 hold (MetalError, GraphFileError). E.3 holds. E.6 holds
(RAII). E.12 n/a. E.14 holds (purpose-designed errors). E.16 holds. E.25-
E.31 n/a.

**Con** — Con.1-Con.5 hold (const locals; `record()` const).

**T** — T.1, T.10 hold for the reader's member-pointer lambda; T.40-T.144
n/a.

**CPL** — n/a.

**SF** — SF.1, SF.2 hold. SF.5 holds. SF.7 holds (`using namespace` only in
.metal kernels' files, as the codebase's). SF.8, SF.12 convention. SF.9
holds (no cycle: warp → nothing; bsdf → warp; target → bsdf; restir_di →
all). SF.10, SF.11 hold (each header compiles alone; the build compiles
every one).

**SL** — SL.1 holds (`std::ranges`). SL.con.1 convention. SL.con.3 holds
(`at()` in tests, bounds checked). SL.str.1 holds. SL.io n/a.

**A** — A.1-A.4 hold (the core's rules in `serenity_core`, the backend's in
`serenity_metal`; `make check` holds the boundary).

**NR, NL, GSL, Tools, FAQ, Discussion** — NL.1-NL.26 hold (names say what;
comments say why; no Hungarian). Discussion (return by value) holds.
Others n/a.

### Performance guidelines (listed fresh: 13 categories, 157 rules)

**cache-layout** — CACHE.1 n/a. CACHE.2 holds (layouts from the passes'
reads). CACHE.3 holds (flat arrays). CACHE.4 holds (a record read whole).
CACHE.5 holds (asserted layouts). CACHE.6 holds (record apart from path
state). CACHE.7 n/a (GPU). CACHE.8 left to the counters (GPU.10).

**codegen** — GEN.1-GEN.8 n/a (no branch hints; shader compiler's).

**concurrency** — CONC.1-CONC.8 n/a (no CPU threads; Film's one atomic,
aggregated per SIMD group, reused).

**copy-move** — COPY.1-COPY.3 n/a. COPY.4, COPY.5 hold (`ReuseImages`
non-copyable). COPY.6 holds (trivially copyable asserted for every shared
layout and probe record). COPY.7-COPY.9 n/a.

**cpu-dsa** — CDSA.1-CDSA.22 n/a. CDSA.21 holds in spirit (tolerances
named). CDSA.23 holds (the reservoir: small state, fixed order, the
softmax test). CDSA.24, CDSA.25 n/a (uniform selection). CDSA.26-CDSA.30
n/a. CDSA.31 n/a (GDSA.18 is the rule cited). CDSA.32, CDSA.33 n/a.

**embedded** — EMB.1-EMB.8 n/a.

**gpu** — GPU.1 holds (nothing read back but the tests' readbacks). GPU.2
holds (row-major, one thread a pixel). GPU.3 A7; the path state written
before the candidates. GPU.4 holds (no-surface pixels return early: a SIMD
group straddling the sky's edge diverges once, as the path pass's). GPU.5
n/a. GPU.6 holds (two dispatches in the frame's encoder). GPU.7 holds (the
reuse constants a ring per frame in flight). GPU.8 A15. GPU.9 A16. GPU.10
open: no frame time measured here; left for an idle GPU.

**gpu-dsa** — GDSA.1 n/a. GDSA.2 holds (fixed stream order). GDSA.3 holds
(the candidates' stream at 2^24, the path's continued from its counter).
GDSA.4 n/a. GDSA.5 holds. GDSA.6 holds (bytes counted in the headers;
Phase A writes 92 a pixel, reads 92). GDSA.7 n/a. GDSA.8 holds. GDSA.9-
GDSA.15 n/a. GDSA.16 holds as the design decided (the shade a pass of its
own). GDSA.17 n/a. GDSA.18 A14. GDSA.19, GDSA.20 n/a. GDSA.21 holds
(softmax by the reservoir, tested).

**lifetime** — LIFE.1 holds. LIFE.2-LIFE.8 n/a.

**memory** — MEM.1-MEM.3 n/a. MEM.4 n/a in Phase A (no array crosses a
frame without the temporal merge). MEM.5-MEM.8 n/a. MEM.9 A16. MEM.10,
MEM.11 n/a.

**simd** — SIMD.1-SIMD.8 n/a (no gathers yet: the spatial merge's).

**telemetry** — TLM.1-TLM.11 n/a.

**wasm** — WASM.1-WASM.14 n/a.

## Counts

| | Findings | Fixed | Kept, stated | Convention | Holds, recorded |
|---|---|---|---|---|---|
| C++ Core Guidelines | 12 (A1-A6, A8-A13) | 5 (A1-A5) | 2 (A6, A13) | 1 (A8) | 4 (A9-A12) |
| Performance | 4 (A7, A14-A16) | 0 | 1 (A7) | 0 | 3 (A14-A16) |

Conventions met (AGENTS.md): SF.8, SF.12, ES.23, ES.27, ES.49, ES.64,
SL.con.1, Enum.7, Enum.8, ES.100, ES.102, ES.107.

## The review of f86e7ae (2026-10-11)

*Codex reviewed f86e7ae twice, for architecture and for performance
(`.cache/reviews/f86e7ae-{architecture,performance}.json`, not in the
repository): seven findings, two of them the same packed-BSDF defect seen
by both. Answered on branch `phase-a-review` from b679fe0 by 0bbb5af (the
structure and the loads), b4ba342 (the packed Bsdf) and c33f186 (the
shade's unpacking, measured). Both corpora
answered: the C++ Core Guidelines through its MCP server, the performance
guidelines through its HTTP endpoint (the `cpp-perf-guidelines` MCP client
was not connected; the endpoint was). Every category of each was listed
and walked over the diff from b679fe0; every finding has an ID and a
verdict.*

### The review's findings

| # | Review | Finding | Rule | Verdict |
|---|---|---|---|---|
| B1 | architecture P1 | `metal/reuse/unbiased.metal.h` included `core/passes/restir_di.h` for `passes::BiasCorrection`; so did `temporal`, `spatial` and `similar.metal.h` for `passes::RestirDi`, and the Sample-reuse-owned reuse constants for the settings | SF.9, A.4 (no cycle, and dependencies one way, between families: change-axes.md); I.4 | fixed: the settings, their three modes and their ranges are Sample reuse's contract, `core/contracts/reuse_settings.h` (`ReuseSettings`, the same 48 bytes); `core/passes/restir_di.h` consumes it (`passes::RestirDi` names it) and keeps the written values. `tools/check_boundaries.sh` rule 5 forbids it again, with probes; on b679fe0's tree it fires on the four reuse headers |
| B2 | architecture P2 | The Pass file defined `first_surface()`, the candidates' transport, `direct_from()` and `shade()` | F.2 (one operation per function, one reason per file: change-axes.md); SF.9 | fixed: three ways. Integrator, `metal/integrator/restir_di.metal.h`: `first_surface()`, `draw_candidate()`, `visible()`, `unshadowed()`, `continue_from()`, seeing reservoirs as contract 14 only. Sample reuse: streaming, finishing, merging, and the stored-test rule (`metal/reuse/visibility.metal.h`, `takes_test()`, `tested_here()`). Pass, `metal/passes/restir_di/restir_di.metal.h`: `candidates()` and `shade()` compose, drawing the numbers and choosing which family each call goes to. Rule 5 holds both families to it |
| B3 | both, P2 | `PackedBsdf` took an ior above 65504 to infinity, and raised positives below 2^-24 to 2^-24, outside its stated relative tolerance | ES.46; I.5, E.7 (state the precondition); P.7 (catch it early) | fixed: contract 2 states a precondition, every packed value finite and at most 65504 in magnitude, and a tolerance in two regimes: 2^-11 relative from 2^-14 up, 2^-24 absolute below. `core/scene/packed_range.h` finds the first material value outside it, naming the material; the Metal renderer refuses a ReSTIR graph over that scene by MetalError, as it refuses 0xffff media. The scene reader and the path graph unchanged |
| B4 | performance P1 | The shade unpacked its record twice, in `direct_from()` and for the continuation | GDSA.18 (decode once where loaded); GPU.3; P.9; Per.6 | measured, and set aside: 0bbb5af unpacked once, as asked; that build's RIS graph measured 0.17 ms (0.5%) slower than the same tree unpacking again for the continuation (`2026-10-10-restir-di-layout.md`, "The review of f86e7ae"). The review's premise, that the RIS graph has no query in the shade, holds at run time but not in the code: the query is compiled into the shade, and the unpacked surface alive across it costs registers in every graph. c33f186 unpacks where each use starts (`continue_from()` takes the record); the headers state the trade, measured |
| B5 | performance P1 | Each candidate read its light's record, sphere, transform and glow three times (draw, evaluate, pdf), and the draw the transform twice; the header's count omitted the pdf | GDSA.6 (count the passes over memory); ES.3; Per.6 | fixed: `place_light()` (`metal/lights/emitter.metal.h`) reads them once into a `PlacedLight`, which `draw_light_point()`, `evaluate_light_point()` and `light_pdf()` read. The header now counts what the code loads at the source: (M + 1) x 76 = 684 bytes a pixel for the RIS graph, (M + 2k + 3) x 76 = 988 with the merges, 13 placements; what the compiled code loads is the counters' |
| B6 | performance P1 | "at most 2.8 ms at 400 GB/s" is a floor on time, not a ceiling | Per.6; GPU.10; GDSA.6 | fixed: reworded as an ideal-bandwidth floor in `restir_di.metal.h`, and in the two other places the same reading appeared, `metal/frame/frame_images.h` (0.6 ms) and `docs/research/2026-10-10-restir-di-layout.md`; `2026-10-09-pass-costs.md` too |
| B7 | performance cost walk | The shade's comment traded registers for flops across a query the RIS graph never traces | GPU.3; GPU.10 | fixed with B4: the comment now says why the trade applies in a graph that traces no shade ray, and cites the measurement; the merges' graphs, where the shade traces, are to measure it again |

No other finding is in either review: the cost walk's remaining items hold
(one dispatch each, two barriers, no readback, no per-light loop, every
helper inline), and its intermediate-traffic arithmetic agrees with the
header's.

### What the code computes, unchanged

- **The path pass, against main 4b61224.** `graphs/path.toml`, rendered
  as before (160 x 96, 3 frames of 4 samples, the four scenes at
  `--time 0`, `8`, `120` and `--step 0.5 --first 3`, `--format pfm`,
  release builds of 0bbb5af, 26314b1 (b4ba342 before an amend to an error
  message) and c33f186): all 48 PFMs byte-identical to main's.
- **The RIS graph, against b679fe0, under safe math.** The same builds
  with `-fmetal-math-mode=safe` set temporarily in
  `cmake/MetalLibrary.cmake`: all 48 `graphs/restir_ris.toml` PFMs
  byte-identical, and all 48 path PFMs. The draw, the evaluation and the
  density read the same values in and compute the same arithmetic; only
  where they load them changed.
- **Under fast math**, the build that ships: 45 of the 48 RIS PFMs differ
  from b679fe0's. 28 of 737,280 values in the first frames differ by more
  than 10^-5 relative, 67 in the third; the largest single value by 22%, a
  4-sample pixel where a rounding flipped a reservoir's choice or a
  roulette; no image's mean by more than 2.3 x 10^-6. Rounding, as for
  b679fe0 itself against f86e7ae.
- **Frame time**, three alternated rounds on the marbles at t = 120
  (`2026-10-10-restir-di-layout.md`, "The review of f86e7ae"): the RIS
  graph 37.15 ms at b679fe0, 36.32 at b4ba342, 36.17 at c33f186; the path
  graph 29.37 to 29.41 at b679fe0, 29.40 to 29.42 at c33f186, unchanged.

### Every category, rule by rule, over the diff from b679fe0

Each rule: **B#** (the finding above), **holds** with where, **convention**
(AGENTS.md "Guideline deviations"), or **n/a** with why.

#### C++ Core Guidelines (listed fresh: In, P, I, F, C, Enum, R, ES, Per, CP, E, Con, T, CPL, SF, SL, A, NR, NL, GSL, Tools, FAQ, Discussion)

**In.0** n/a.

**P** — P.1 holds (the three families named in code: one header each, the
Pass calling across). P.2 holds for host code; MSL by decision. P.3 holds
(`place_light`, `draw_candidate`, `unshadowed`, `continue_from`,
`takes_test`, `outside_packed_bsdf`). P.4 holds (layouts asserted,
`ReuseSettings` 48 bytes). P.5 holds (a light kind added fails each
`PlacedLight` switch's build). P.6, P.7 B3 (the precondition checked when
the renderer is built, before any frame). P.8 n/a. P.9 B4, B5. P.10 holds.
P.11 n/a. P.12 B1 (the boundary a tool's, proven by probes). P.13 n/a.

**I** — I.1 holds. I.2 holds (no globals). I.3 n/a. I.4 B1 (the modes enum
classes in the contract); `PlacedLight`, `Weighed`, `Unshadowed` named
types. I.5 B3; `draw_candidate`, `unshadowed`, `evaluate_light_point`
state theirs. I.6, I.8 n/a in MSL; the host's precondition thrown. I.7
holds (`outside_packed_bsdf` says what it returns). I.10 holds
(MetalError). I.11 n/a. I.13 n/a. I.22 holds. I.23 holds (at most six
parameters: `continue_from`, `shade`). I.24 holds (no two adjacent
parameters of one type in the new functions: `choose` a float beside the
float2 `point_numbers`). I.25-I.30 n/a.

**F** — F.1 holds. F.2 B2. F.3 holds (`candidates` 20 lines, `shade` 25,
`outside_packed_bsdf` 40). F.4, F.5 hold (shader functions inline). F.6
n/a. F.7 holds. F.8 holds (`place_light` reads only its arguments' arrays;
the packing pure). F.9 n/a. F.10 holds. F.15, F.16 hold (small structs by
value in MSL; `const SceneDescription&` on the host). F.17 holds
(`numbers` in-out, as before). F.20, F.21 hold (`Weighed`, `Unshadowed`,
`Shaded` returned). F.42-F.55 n/a. F.56 holds (`shade` two levels deep).

**C** — C.1, C.2 hold (`PlacedLight`, `Weighed`, `Unshadowed` structs of
independent members). C.162 holds (`evaluate_light_point` overloaded on
`Lights` and `PlacedLight`, the same operation; `light_pdf` likewise).
C.181 holds (`PlacedLight` holds a sphere's fields beside its record, not
a union; a light kind added adds its own). Others n/a.

**Enum** — Enum.3 holds. Enum.7, Enum.8 convention (the moved modes are
shared with Metal). Others hold or n/a.

**R** — n/a (no ownership added).

**ES** — ES.3 B5 (one placement), B2 (one `takes_test`), and the
transport still shared with the path tracer. ES.5, ES.6 hold. ES.12 holds
(-Wshadow; `outside` the local in the renderer, `base_outside` the
helper). ES.20 holds (`PlacedLight placed{}`, `Weighed{...}`). ES.23
convention in MSL. ES.45 B3 (`packed_largest`, `packed_least_normal`,
`packed_relative`, `packed_absolute`). ES.46 B3. ES.48, ES.49 hold. ES.70,
ES.79 hold (switches on kinds, no default). ES.100, ES.107 convention.
ES.105 holds (no new division). Others n/a.

**Per** — Per.1-Per.4 hold (B4 and B5 measured, above; B4 set aside on
the measurement). Per.6 B6, and
each claim of this change measured (above). Per.7 holds. Per.19 holds.
Others n/a.

**CP** — n/a.

**E** — E.2 holds (MetalError). E.7 B3. E.14 holds. Others n/a.

**Con** — Con.1-Con.4 hold (const locals and parameters).

**T** — n/a (no templates added).

**CPL** — n/a.

**SF** — SF.1, SF.2 hold. SF.5 holds. SF.7 holds (no `using namespace` in a
header). SF.8 convention. SF.9 B1, B2 (`reuse_settings.h` includes
`reservoir.h`, which only names it in a comment). SF.11 holds (each new
header includes what it uses). SF.12 convention. Others n/a.

**SL** — SL.str.1 holds (`std::string` for messages). SL.io.50 n/a.
SL.con.3 holds (`.at()` on the description's arrays). Others n/a.

**A** — A.4 B1 (as SF.9, between families). Others n/a.

**NR, NL, GSL, Tools, FAQ, Discussion** — NL.1-NL.26 hold (names say what:
`place_light`, `tested_here`, `material_names`). Others n/a.

#### Performance guidelines (listed fresh: 13 categories, 157 rules)

**cache-layout** — CACHE.1 n/a. CACHE.2, CACHE.3 hold (the light arrays as
before, read once a candidate). CACHE.4-CACHE.8 n/a (no layout changed;
`ReuseSettings` is `RestirDi`'s layout).

**codegen** — GEN.1-GEN.8 n/a (the shader compiler's; every helper
`inline`, as before).

**concurrency** — n/a.

**copy-move** — COPY.6 holds (`ReuseSettings` asserted trivially
copyable). Others n/a.

**cpu-dsa** — CDSA.31 holds in spirit (B3: a quantized form used only
where the value fits it, refused where it does not). CDSA.32 n/a. Others
n/a.

**embedded** — n/a.

**gpu** — GPU.1, GPU.2 hold. GPU.3 B4, B7. GPU.4 holds (the shade's
branches as before). GPU.5-GPU.9 hold or n/a (no dispatch, barrier or
resource changed). GPU.10 B6; frame times measured (above); counters not
recorded (2026-10-11-path-split-cost.md: `xctrace` saves one recording in
twelve).

**gpu-dsa** — GDSA.2, GDSA.3 hold (the candidates' numbers drawn in the
same order). GDSA.6 B5, B6. GDSA.16 holds as decided. GDSA.18 B4.
Others n/a.

**lifetime, memory, simd, telemetry, wasm** — n/a (no allocation, lifetime,
SIMD path, telemetry or WebAssembly touched).

### Counts

| | Findings | Fixed | Set aside on measurement | Convention |
|---|---|---|---|---|
| The review's | 7 (B1-B7) | 6 (B1-B3, B5-B7) | 1 (B4) | 0 |
| Found by the sweep beyond them | 0 | 0 | 0 | 0 |

Conventions met (AGENTS.md): SF.8, SF.12, ES.23, Enum.7, Enum.8, ES.100,
ES.107.
