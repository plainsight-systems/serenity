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
