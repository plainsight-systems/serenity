# ReSTIR DI's headers against the two guideline corpora, rule by rule

*2026-10-10. The headers designed on branch `restir-di-headers`, from
1a6b78f. Both corpora answered: the C++ Core Guidelines through its MCP
server, and the performance guidelines through its HTTP endpoint
(localhost:7015; its MCP client was not connected, the endpoint was). Every
category of each was listed fresh and walked; every finding has an ID and a
verdict, and every valid one is fixed in the same commit.*

## Scope

New: `core/passes/restir_di.h`; `core/contracts/light_sample.h`,
`surface_record.h`, `reservoir.h`, `reuse_constants.h`;
`metal/reuse/reservoir.metal.h`, `unbiased.metal.h`, `similar.metal.h`,
`temporal.metal.h`, `spatial.metal.h`; `metal/integrator/target.metal.h`,
`restir_di.metal.h`; `metal/passes/{surfaces, restir_candidates,
restir_temporal, restir_spatial, restir_shade}/*.h`.

Added to: `core/frame/schedule.h`, `graph_file.h`, `history.h`,
`frame_inputs.h`; `metal/integrator/path.metal.h`;
`metal/lights/sphere_light.metal.h`, `emitter.metal.h`;
`metal/light_selection/uniform_light.metal.h`; `metal/sampler/
sampler.metal.h`; `metal/camera/thin_lens.metal.h`; `metal/math/
warp.metal.h`; `metal/passes/bindings.h`; `metal/frame/frame_resources.h`,
`frame_images.h`, `accumulation.h`, `renderer.h`;
`metal/acceleration/scene_acceleration.h`; `metal/scene/
shape_transforms.h`, `light_glows.h`.

Each header was compiled on its own (`xcrun metal -std=metal4.0 -Werror`
for shader headers and, for the shared layouts, both languages; `clang++
-std=c++20 -Wall -Wextra -Werror` with metal-cpp for host headers): all 41
pass, so SF.11 holds. Every existing `.metal` still compiles.

## The IDs the layout cited, checked

| ID | Cited for | Says it? |
|---|---|---|
| CDSA.23 | the reservoir: small mergeable state, fixed combine order | yes |
| GDSA.21 | picking by w / w_sum is the random-key race's draw | yes, as the same distribution (the headers say so; not the same draw for the same numbers) |
| GDSA.6 | counting the passes over memory | yes |
| GDSA.3 | numbers from pixel, frame and purpose | yes |
| GPU.2 | flat records, one a pixel, coalesced | yes |
| GPU.3 | fewer registers across ray queries, judged by time | yes |
| GPU.7 | lifetimes of the kept structure and the copies | yes |
| GPU.8 | a barrier only between a write and its reader | yes |
| GPU.9 | images made outside the loop, released after the GPU | yes |
| MEM.4 | ping-ponged records and final reservoirs | yes |
| MEM.9 | allocated at init (per size), nothing per frame | yes |
| CACHE.6 | record (hot) apart from path state (cold) | yes |
| CDSA.31 | quantizing the two directions | loosely: its point is quantize, then rerank from fuller precision. GDSA.18 (codes decoded in the consumer's load path) is the rule for the light point and is cited instead (row F14) |
| SIMD.5 | neighbor reads are gathers | yes |
| Enum.3 | PassKind an enum class | yes |
| P.5 | shared layouts' static_asserts | yes |
| P.6 | "a switch with no default, so a backend missing a kind fails its build" | no: that is compile-time checking, P.5 (and ES.79). P.6 fits the run-time refusal (std::logic_error) schedule.h already cites it for. The headers do not repeat the layout's citation (row F13) |
| I.5, I.7 | preconditions, postconditions | yes |
| ES.3 | one target, one visibility test, one light transport | yes |
| ES.45 | settings' values named | yes |
| ES.46 | the 16-bit point's loss stated | yes: a deliberate narrowing, stated as the rule's narrow_cast asks |
| C.2 | settings as a struct | yes: the members vary independently; each range is per member |
| Per.7 | settings as data a measurement varies | yes, loosely: design to enable later optimization |
| F.4 | the target "pure, inline" | no: F.4 is constexpr. F.8 (pure functions) is cited instead (row F12) |
| P.1 | the algorithm stated where implemented | yes |
| I.1 | the identity explicit | yes |
| GDSA.8 | one thread a pixel, by output | yes |
| SL.io.2, E.14 | ill-formed graph input refused, by GraphFileError | yes |

## Findings and verdicts

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| F1 | I.24 | reuse/reservoir.metal.h | stream_candidate(reservoir, y, float target, float density, float u): three adjacent floats | fixed: a `Candidate` struct |
| F2 | I.23 | reuse/reservoir.metal.h | stream_reservoir took six parameters | fixed: a `MergeInput` struct; three parameters |
| F3 | I.23, I.24 | reuse/temporal.metal.h | merge_temporal took nine, two SurfaceRecords and two Reservoirs among them | fixed: `Merging` (unbiased.metal.h, shared with spatial) and `LastFrame`; six, no two of one type adjacent |
| F4 | I.23 | reuse/spatial.metal.h | merge_spatial took ten | fixed: `Merging` and `NeighborArrays`; seven, each of a different type, kept: a bundle of the scene, camera and reuse constants would be a "context" with no meaning of its own |
| F5 | I.24 | sampler/sampler.metal.h | purpose_numbers(uint2, uint frame, uint first): frame and dimension adjacent words | fixed: purpose_numbers(PathNumbers stream, uint first) |
| F6 | I.24, SF.9 | lights/sphere_light.metal.h, emitter.metal.h | sphere_light_at_point(..., float3 on_unit, float3 point): two adjacent positions. And the octahedral map lived in emitter.metal.h, which includes sphere_light.metal.h, so the sphere could not use it without a cycle | fixed: the sphere takes and returns the encoded word; the map moved to metal/math/warp.metal.h, shared shader math |
| F7 | I.24 | integrator/target.metal.h | surface_record(float3 position, float3 wo, Bsdf) | fixed: surface_record(Shading here) |
| F8 | I.24 | acceleration/scene_acceleration.h | update_kept(..., uint32 slot, uint64 index, uint64 sequence, ...) | fixed: the slot and sequence as begin()'s FrameSlot |
| F9 | ES.8, NL.19 | contracts/light_sample.h, lights/*.metal.h | light_at_point() beside the existing light_at() (which light a shape is): similar names, unrelated meanings | fixed: evaluate_light_point(), sphere_evaluate_point(), and the result LightReceived |
| F10 | F.1, NL.8 | core/frame/schedule.h | restir(kind) is no predicate phrase among needs_scene, accumulates, writes_radiance | fixed: is_restir(kind) |
| F11 | F.1 | reuse/unbiased.metal.h | hidden(reservoir) reads as a predicate; it returns a changed reservoir | fixed: hide(reservoir) |
| F12 | F.4, F.8 | integrator/target.metal.h | the layout's F.4 citation (constexpr) for pure functions | fixed: cites F.8 |
| F13 | P.5, P.6, ES.79 | layout | P.6 cited for a compile-time check | recorded here; the headers cite nothing new for it |
| F14 | CDSA.31, GDSA.18, ES.46 | contracts/light_sample.h, surface_record.h | the layout quantized wo too; an f evaluated at a quantized wo is the image of another surface, and its support differs at grazing angles | fixed: wo kept at full precision in the record, in the 8 bytes Bsdf pads; the light point quantized and decoded in the load path (GDSA.18) |
| F15 | GPU.4 | reuse/temporal.metal.h | Z's one ray is against last frame's structure or this frame's by the survivor's source: as a branch, two queries a SIMD group | fixed: one query, its structure, origin and light chosen per lane as data |
| F16 | GPU.4 | reuse/spatial.metal.h | the own-visibility ray beside k neighbor rays: a branch per lane | fixed: one loop of k, the own test taking the survivor's source's iteration |
| F17 | GDSA.16 | integrator/restir_di.metal.h | records, path states and reservoirs are materialized intermediates | fixed: stated and justified in the header (the spatial merge's grid-wide dependency; GPU.3), the surfaces-candidates fusion left to the counters |
| F18 | SF.10 | reuse/spatial.metal.h | CameraData reached through reuse_constants.h's include | fixed: includes core/contracts/camera.h |
| F19 | NL.17 | schedule.h, graph_file.h, sampler.metal.h, bindings.h, unbiased.metal.h | comment lines past the house's wrap | fixed: rewrapped |
| F20 | I.9, T.10 | unbiased.metal.h, temporal.metal.h, spatial.metal.h | templates without concepts | fixed as far as the language allows: the shading language has no concepts, so each template's requirements are stated where it is declared |
| F21 | Per.6 (and the user's rule: no measurements in headers) | path.metal.h, renderer.h, scene_acceleration.h | a frame time (29.90 ms), a CPU cost (43 µs) and a structure's size taken from the layout | fixed: removed; the headers point to the note |
| F22 | ES.45 | core/passes/restir_di.h | range ends 0, 1 and -1 for the thresholds are unnamed | rejected: the domain of a ratio and of a cosine, not tuning numbers; the tuning numbers are named |
| F23 | I.4, Enum.3 | reuse/reservoir.metal.h | Streaming::source is a uint with two named sentinels, not a type | rejected: it also holds a neighbor's index; an enum would need a cast at every neighbor (Enum.8's project deviation for shader-shared values); the sentinels are named |
| F24 | F.16 | shader headers | 64-byte records and the scene passed by value | rejected: the house convention for inline shader functions, kept consistent with path.metal.h; the corpus sweep's verdict stands (no codegen change shown) |
| F25 | GPU.9, MEM.11 | metal/frame/frame_images.h | the working reservoirs and path states live only within a frame and could alias the tone map's pyramid in one heap | rejected for now (Per.1, Per.2): what aliasing would save, 16 + 64 bytes a pixel, has no measured cost to recover, and aliasing adds hazards the renderer must barrier; revisit if memory at the window's traced resolution becomes a constraint |
| F26 | GDSA.15, SIMD.5 | reuse/spatial.metal.h | gathers within 30 pixels in row-major order | kept, stated: a space-filling order is the follow-up if the counters show the spatial pass bound by the lines it touches (GPU.10) |

## Every category, rule by rule

Notation: **holds** (applies, and the headers keep it), a finding number,
**convention** (AGENTS.md "Guideline deviations"), or **n/a** with why.

### C++ Core Guidelines (listed fresh: In, P, I, F, C, Enum, R, ES, Per, CP, E, Con, T, CPL, SF, SL, A, NR, NL, GSL, Tools, FAQ, Discussion)

**In.0** n/a (an introduction).

**P** — P.1 holds (each algorithm by numbered steps where it is carried
out). P.2 holds for host headers; the shader headers are MSL, by decision
(MEMORY.md). P.3 holds (intent in every type and name; F9-F11). P.4 holds
(shared layouts asserted; no casts declared). P.5 holds (sizes and trivial
copyability asserted; a new PassKind fails every switch; F13). P.6 holds
(schedules, settings and histories checked at run time, refused by name).
P.7 holds (all checked before the submission begins: renderer.h). P.8
holds (every GPU object a Resident; ReuseImages and the kept structure
released after the GPU). P.9 holds (costs counted; nothing recomputed that
a pass already has, but the spatial merge's second walk, chosen over 64 k
bytes of registers: spatial.metal.h). P.10 holds (records and reservoirs
immutable after their pass but where a pass's contract says in place).
P.11 holds (the octahedral map and the visibility test each in one place;
F6). P.12, P.13 n/a (no tools or libraries added).

**I** — I.1 holds (identity explicit; preconditions named). I.2 holds (no
globals). I.3 holds. I.4 holds (Keep, Temporal, Working, ReservoirOutput,
FirstLight are types; F23 rejected). I.5 holds (each precondition stated:
candidates, evaluate_light_point, continue_path, merge_temporal). I.6, I.8
n/a in MSL (no Expects/Ensures); host preconditions are checked and thrown,
as the codebase does. I.7 holds (unbiased.metal.h's postcondition;
history.h's). I.9 F20. I.10 holds (failures thrown: HistoryError,
MetalError, GraphFileError). I.11 holds (no ownership through raw
pointers; device arrays are views). I.12 n/a (no not_null in the
codebase). I.13 holds (NeighborArrays carries its size with its arrays).
I.22 holds. I.23 F2, F3, F4. I.24 F1, F3, F5-F8. I.25-I.27 n/a (no
hierarchies, no ABI). I.30 holds (the shared-normal offset, a stated
limit, is encapsulated in one function, restir::visible).

**F** — F.1 F10, F11, else holds. F.2 holds (each declaration one
operation; merge_temporal's six steps are one merge). F.3 n/a (no bodies;
each step a function). F.4 n/a (shader functions; host functions do I/O or
throw). F.5 holds (shader functions inline). F.6 holds (addresses()
noexcept; the throwing functions are not). F.7 holds (references, not
smart pointers, in every parameter). F.8 holds (target, similarity,
reprojection pure; F12). F.9 holds. F.10 holds (visible(), supports(),
similar() named and shared). F.15 holds. F.16 F24. F.17 holds (the
reservoir and the numbers by thread&). F.18, F.19 n/a. F.20 holds (results
returned: Surfaces, Reprojected, Neighbor, Traced, Addresses). F.21 holds
(struct returns). F.22 holds. F.23 n/a. F.24 holds where host spans are
taken (transforms). F.25 n/a. F.26, F.27 n/a. F.42-F.49 hold (no pointer
or reference returns, no const returns). F.50-F.54 n/a (no lambdas; MSL).
F.55 holds. F.56 n/a (no bodies). F.60 holds (last_transforms a pointer,
null without a temporal merge, stated).

**C** — C.1 holds (Merging, LastFrame, NeighborArrays, Candidate,
MergeInput, Segment). C.2 holds (structs for data, classes for
ReuseHistory, LiveReuse, ReuseImages, the passes). C.3, C.4 hold. C.5
holds. C.7 holds. C.8 holds. C.9 holds (private members). C.10 holds (no
hierarchies). C.11 n/a (no regular types defined beyond aggregates). C.12
holds (Submission& members are in non-copyable, non-movable classes, as
the codebase does). C.13 holds (ReuseImages' members in dependency
order). C.20, C.21 hold (copies and moves deleted together, destructor
defaulted). C.22 holds. C.30-C.37 hold (no destructor needed: Residents
release). C.40-C.42 hold (constructors throw if they cannot make their
object). C.43 n/a (non-copyable). C.44 n/a. C.45 holds (default member
initializers in ReuseHistory, LiveReuse, Sample, HeadlessRun, Addresses).
C.46 holds (single-argument pass constructors explicit). C.47 n/a. C.48
holds. C.49 n/a. C.50-C.52 n/a. C.60-C.67 n/a (no copy or move defined).
C.80 holds (= default destructors). C.81 holds (= delete). C.82-C.90 n/a
(no virtual functions, swap, ==, hash, memset). C.100-C.109 n/a (no
containers). C.120-C.153 n/a (no hierarchies). C.160-C.170 n/a (no
operators). C.180-C.183 n/a (no unions).

**Enum** — Enum.1 holds (constants, not macros). Enum.2 holds. Enum.3
holds (every new enum a class enum, but bindings: convention). Enum.4 n/a.
Enum.5 holds. Enum.6 n/a. Enum.7, Enum.8 convention (bindings' plain
enums with values, the binding layout).

**R** — R.1 holds (Residents). R.2, R.3, R.4 hold (raw pointers and
references non-owning). R.5 holds. R.6 holds. R.10-R.15 n/a (no
allocation in headers). R.20-R.24 n/a (no new ownership; members follow
the codebase's NS::SharedPtr). R.30-R.37 n/a.

**ES** — ES.1 holds where the library has it. ES.2 holds. ES.3 holds
(one target, one visibility test, one similarity test, one light
transport). ES.5, ES.6 n/a (no bodies). ES.7 holds (short names local,
long names wide). ES.8 F9. ES.9 holds. ES.10 holds. ES.11, ES.12 n/a.
ES.20 holds (host aggregates initialized; shader structs made by named
factories: empty_reservoir()). ES.21, ES.22 n/a. ES.23 holds. ES.24 n/a.
ES.25 holds (constants constexpr). ES.26 holds (each array one purpose;
the working reservoirs' in-place merge is stated). ES.27 convention
(padding arrays in shared layouts). ES.28 n/a. ES.30-ES.33 holds
(SERENITY_CONSTANT is the codebase's one macro, unchanged). ES.34 holds.
ES.40, ES.41 n/a. ES.42 holds. ES.43, ES.44 n/a. ES.45 holds (every
setting and sentinel named; F22 rejected). ES.46 holds (the 16-bit
encoding stated; M exact as a float, stated). ES.47 holds (null
last_transforms). ES.48-ES.50 holds (no casts declared; MSL's conversion
syntax a convention). ES.55 holds (indices bounded by the image size,
stated). ES.56 n/a. ES.60-ES.65 n/a. ES.70-ES.87 n/a (no statements),
but ES.79 holds for the switches the new kinds join. ES.100-ES.107
convention (uint in shaders) and hold on the host (uint64 indices, as
history.h has).

**Per** — Per.1, Per.2 hold (no optimization without a measurement plan;
F25). Per.3 holds. Per.4, Per.5 n/a. Per.6 F21; holds now (every number
a derivation, said to be one). Per.7 holds. Per.10 holds. Per.11 holds
(settings checked and laid out on the CPU). Per.12, Per.13 hold (a light
by its record's index, no further indirection). Per.14, Per.15 hold
(nothing allocated per frame). Per.16 holds (16-byte reservoirs, 64-byte
records). Per.17 holds (position first: the target reads it first). Per.18
holds (bytes counted). Per.19 holds (row-major, own-pixel reads coalesced;
gathers counted). Per.30 n/a.

**CP** — CP.1-CP.111 and CP.200-CP.201: n/a. No CPU threads are added;
the GPU's concurrency is the frame graph's barriers (GPU.7, GPU.8), and
the one CPU wait added (update_kept's) is for a submission it needs
(CP.42 holds in spirit: it waits on a condition, the sequence).

**E** — E.1 holds (the codebase's: exceptions on the host, refusals by
name). E.2 holds. E.3 holds. E.4 holds (ReuseHistory changes nothing when
it refuses). E.5 holds. E.6 holds. E.7, E.8 hold (stated). E.12-E.19
holds or n/a (no new handlers). E.14 holds (GraphFileError, HistoryError,
MetalError). E.25-E.28 n/a. E.30, E.31 n/a.

**Con** — Con.1 holds. Con.2 holds (SeenFrom's questions const; the
acceleration's traced() const; ReuseImages::addresses() const). Con.3
holds (const device pointers for read-only arrays). Con.4, Con.5 hold.

**T** — T.1, T.2 hold (the merges generic over the scene type, so either
frame's scene is passed alike). T.10, T.11, T.13 F20 (no concepts in MSL).
T.12 n/a. T.20-T.26 n/a. T.40 holds (SeenFrom is a function object for
supports()). T.41 holds (supports() requires target() and visible() only).
T.42-T.44 n/a (aliases: Scene by `using`, T.43 holds). T.47 holds
(names specific). T.48 n/a. T.49 holds. T.60 holds. T.61-T.84 n/a.
T.100-T.103 n/a. T.120-T.125 n/a. T.140, T.141 hold or n/a. T.142-T.150
n/a.

**CPL** — CPL.1-CPL.3 n/a.

**SF** — SF.1 holds (.h, .metal.h). SF.2 holds (only inline declarations
and constexpr constants). SF.3 holds. SF.4 holds. SF.5 n/a (no .cpp).
SF.6, SF.7 hold (no using namespace in headers). SF.8 convention. SF.9
F6. SF.10 F18. SF.11 holds (each header compiled alone). SF.12
convention. SF.13 holds. SF.20 holds (shaders::restir, metal, frame,
passes, contracts). SF.21, SF.22 hold.

**SL** — SL.1, SL.2 hold. SL.3 holds. SL.4 holds. SL.C.1 n/a. SL.con.1
convention (shared layouts' arrays; host members are std::array). SL.con.2
n/a. SL.con.3 holds (sizes carried with arrays). SL.con.4 holds (trivially
copyable asserted). SL.io.1-SL.io.50 n/a, but SL.io.2 holds for the graph
reader's new table. SL.str.* n/a.

**A** — A.1 holds (contracts stable, shaders changeable). A.2 n/a. A.4
holds (no cycles: F6).

**NR** — NR.1-NR.7 n/a (none of the myths is followed).

**NL** — NL.1 holds (comments say what code cannot: contracts, costs,
why). NL.2 holds (intent stated). NL.3 holds as the codebase writes headers. (NL.2's truth was checked
against RTXDI's resampling sources and the layout's statement of the
paper's algorithms; the paper's own text was not at hand.)
NL.4 holds. NL.5 holds. NL.7 holds. NL.8 F10. NL.9 holds. NL.10 holds.
NL.11 holds (2^24 as `1u << 24`). NL.15-NL.18 hold; NL.17 F19. NL.19 F9.
NL.20, NL.21 hold. NL.25-NL.27 hold.

**GSL** — GSL.ptr n/a. **Tools**, **FAQ.***, **Discussion** n/a.

### Performance guidelines (listed fresh: 13 categories, 157 rules)

**cache-layout** — CACHE.1 n/a (no CPU sharing). CACHE.2 holds (layouts
from what the target reads). CACHE.3 holds (flat arrays, no pointers).
CACHE.4 holds (AoS: a record's fields read together). CACHE.5 holds
(groups of a triple and a word). CACHE.6 holds (record and path state
split). CACHE.7 n/a (GPU). CACHE.8 holds by deferral to the counters
(GPU.10).

**codegen** — GEN.1-GEN.8 n/a (shader compiler; host headers add no hot
code).

**concurrency** — CONC.1-CONC.8 n/a (no CPU threads; no atomics but Film's
existing counter).

**copy-move** — COPY.1-COPY.3 n/a. COPY.4 holds (ReuseImages: rule of
zero but for deletions stated). COPY.5 holds (non-copyable owners). COPY.6
holds (trivially copyable asserted). COPY.7 n/a. COPY.8 holds (factories
return by value). COPY.9 n/a.

**cpu-dsa** — CDSA.1-CDSA.22 n/a (no CPU structures; CDSA.21's injectable
thresholds hold in spirit: the similarity thresholds are the graph's).
CDSA.23 holds. CDSA.24, CDSA.25 n/a (uniform selection; power-weighted
selection is a later, measured change). CDSA.26-CDSA.30 n/a (the
structure's rebuild is unchanged; CDSA.29, CDSA.30 hold as before).
CDSA.31 see F14. CDSA.32 n/a. CDSA.33 n/a.

**embedded** — EMB.1-EMB.8 n/a.

**gpu** — GPU.1 holds (nothing read back). GPU.2 holds. GPU.3 holds.
GPU.4 F15, F16. GPU.5 n/a now (threadgroup memory for the spatial gathers
would need a halo of the radius; measured first). GPU.6 holds (five
dispatches in the frame's one encoder). GPU.7 holds (the kept structure's
ring; last frame's copies). GPU.8 holds (each barrier named). GPU.9 holds;
F25. GPU.10 holds (the spatial pass's time first).

**gpu-dsa** — GDSA.1 n/a. GDSA.2 holds (each reservoir's sums in a fixed
order: deterministic). GDSA.3 holds. GDSA.4 n/a (no cross-group waits).
GDSA.5 holds (Film's counter, aggregated per SIMD group, reused). GDSA.6
holds. GDSA.7 n/a. GDSA.8 holds. GDSA.9-GDSA.14 n/a. GDSA.15 F26.
GDSA.16 F17. GDSA.17 n/a. GDSA.18 holds (F14). GDSA.19, GDSA.20 n/a.
GDSA.21 holds.

**lifetime** — LIFE.1 holds (trivially copyable layouts). LIFE.2-LIFE.8
n/a (FrameArray's view<T> is the one reinterpretation, unchanged).

**memory** — MEM.1-MEM.3 n/a. MEM.4 holds. MEM.5 n/a. MEM.6 n/a (the
codebase allocates through the device). MEM.7, MEM.8 n/a. MEM.9 holds.
MEM.10 n/a. MEM.11 F25.

**simd** — SIMD.1-SIMD.4 n/a (CPU vectorization). SIMD.5 holds (gathers
counted; F26). SIMD.6-SIMD.8 n/a.

**telemetry** — TLM.1-TLM.11 n/a (no telemetry added).

**wasm** — WASM.1-WASM.14 n/a.

## Counts

| | Findings | Fixed | Recorded (a citation in the layout) | Rejected, with reason | Kept, stated |
|---|---|---|---|---|---|
| C++ Core Guidelines | 20 | 16 (F1-F12, F18-F21) | 1 (F13) | 3 (F22, F23, F24) | 0 |
| Performance (F14 also cites ES.46) | 6 | 4 (F14-F17) | 0 | 1 (F25) | 1 (F26) |
| All | 26 | 20 | 1 | 4 | 1 |

Conventions met (AGENTS.md): SF.8, SF.12, ES.27, SL.con.1, Enum.7, Enum.8,
ES.100, ES.102, ES.107, and MSL's conversion syntax.

## The review of 70dbe59

Codex reviewed 70dbe59 for performance and for architecture
(`.cache/reviews/70dbe59-performance.json`, `70dbe59-architecture.json`
in the main checkout). Every finding was valid and is fixed in the commit
after it. The IDs below were checked against the corpora before they were
cited: GPU.6 (fuse stages), GDSA.6, GDSA.16, GPU.2, CACHE.5, GPU.7, MEM.4,
SL.con.3, ES.3, P.5 and I.24, each fetched or listed and read for what it is
cited for.

| # | Review | Finding | IDs | What changed |
|---|---|---|---|---|
| R1 | both, P1 | The 16-bit octahedral light point makes the estimator a quadrature over grid cells, biased, which F14 above had stated as small instead of removing | ES.46 | contracts/light_sample.h: the point stored as drawn, three floats in the light's own space; LightPoint 16 bytes. The octahedral map, its "not zero" statement and its warp.metal.h declarations removed; nothing else used them. GDSA.18 and ES.46 no longer apply: nothing is narrowed. F14's verdict is superseded |
| R2 | both | The reservoir's size with an exact point | GPU.2, CACHE.5 | contracts/reservoir.h: 32 bytes, the 24 of its fields and 8 of named padding, so it never straddles a cache line and loads as two aligned 16-byte words; why stated there |
| R3 | performance, P1 | Surfaces and candidates split for no dependency: 80 bytes a pixel written and read back, a dispatch and a barrier | GDSA.6, GPU.6 | One pass, `restir_initial` (steps 1 to 3); the path state written before the candidate loop. schedule.h's kinds, order rules and graphs (RIS alone is ["restir_initial", "restir_shade"]), the pass directory and header, bindings.h, renderer.h and every cost table follow |
| R4 | architecture, P1 | The Integrator's restir_di.metal.h included Sample reuse | change-axes.md | Moved to metal/passes/restir_di/restir_di.metal.h, a Pass-axis composition the four passes include. The merges ask a Frame (unbiased.metal.h states it: target, visible, depth) that the composition's FrameView provides, so metal/reuse/ includes only contracts, the settings, its own headers and the Sampler; the Integrator keeps target.metal.h and the path split and includes nothing of Sample reuse. Each header states its dependencies |
| R5 | architecture, P2 | SurfaceRecord copied Bsdf's fields, a second definition of Material's layout | ES.3 | The record embeds contract 2's Bsdf whole, with position and wo at full precision and two words of padding: 80 bytes. record_bsdf() removed |
| R6 | architecture, P1 | `first_written` inserted mid-aggregate broke plan_headless()'s positional initialization: a still run's samples became 0 | I.24 (as for parameters, so for positional aggregates) | HeadlessPlan::first_written, Sample::reused_since, HeadlessRun::reuses and FrameInputs::reused_since all declared last, each saying the implementation wires it. Checked: plan_headless({first 3, samples 4, accumulates}) gives 4 samples, sample (3, 2) index 14, and a moving run keeps `instants` (a scratch program against history.cpp) |
| R7 | both, P2 | Three kept structures promised, storage for two | SL.con.3, GPU.7, MEM.4 | scene_acceleration.h: `std::array<Built, frames_in_flight + 1>`, the last entry made only with Keep::last_too and moving shapes, and each entry's last tracing submission, for the one wait update_kept() makes |
| R8 | architecture, P2 | schedule.cpp's assertion held the list to tone_map, so a ReSTIR kind left out of it would compile | P.5 | PassKind gains `last = restir_shade`, a second name for the final value, which needs no switch case (checked with -Wswitch -Werror); schedule.cpp asserts against PassKind::last, and now fails until the implementation lists the four kinds, as its switches do |
| R9 | performance, P1 | The bandwidth total omitted light data and traversal | GDSA.6 | The composition's cost is now three kinds, not added: intermediate-buffer traffic (1368 to 1928 bytes a pixel at k = 5, logical, and DRAM as far as uncached); light data, 76 logical bytes per target evaluation, 1748 a pixel at the written settings, assumed cached, 13.5 GB a frame at the display's size were it not; traversal as the rays' own cost |
| R10 | architecture, P1 | unbiased.metal.h said the spatial mask avoided rereading neighbors; spatial.metal.h rereads them | GDSA.6 | Both now say the mask keeps acceptance only, and the second walk's k x 112 bytes are counted |

What the rule-by-rule pass above says that the fixes change:

- I.24 holds for sphere_evaluate_point(): with the octahedral word gone it
  takes the LightPoint, not a bare vector beside the point (F6's fix,
  kept by other means).
- SF.9 holds without the warp.metal.h map: nothing shared remains for
  sphere_light.metal.h and emitter.metal.h to cycle over.
- I.9 and T.10: the merges are generic over a Frame, stated in
  unbiased.metal.h, as supports() was over a Surface (F20).
- GDSA.16 (F17): the reason the intermediates are written now names the
  one split that stays without a grid-wide dependency to justify it, the
  shade's, which keeps the continuation out of the merges' kernels; steps 1
  to 3 are one kernel.
- Every header compiles on its own again (SF.11), and every existing shader.

New byte counts: SurfaceRecord 80, PathState 64, LightPoint 16, Reservoir
32; per pixel at k = 5, M = 8, all four passes: 1368 to 1928 bytes of
intermediate traffic, 1748 logical bytes of light data; ReuseImages 332
bytes a pixel, 2.56 GB at 3456 x 2234 and 641 MB at 1728 x 1117.

## The production rework

*2026-10-10, on branch `restir-di-production` from 4463eee. The user
directed that ReSTIR DI's default be RTXDI's shipped production defaults,
with the unbiased ray-traced merges kept only as a setting for checking
against the reference (the layout note, "Decided"). Both corpora answered:
the C++ Core Guidelines through its MCP server, every category listed fresh
(In, P, I, F, C, Enum, R, ES, Per, CP, E, Con, T, CPL, SF, SL, A, NR, NL,
GSL, Tools, FAQ, Discussion); the performance guidelines through their HTTP
endpoint (localhost:7015; the `cpp-perf-guidelines` MCP client failed to
connect, the endpoint answered), its 13 categories and 157 rules listed
fresh. Every ID below was listed, and those whose wording a verdict turns
on were fetched and read: GDSA.18, CDSA.21, CDSA.31, GPU.2, GPU.5, CACHE.5,
GDSA.6, ES.46, Per.7, I.4.*

### Scope

Changed: `core/contracts/light_sample.h` (8 bytes, the valid bit, the
equal-area point and its quantization), `reservoir.h` (24 bytes),
`surface_record.h` (a 32-byte record, a 36-byte path state), `bsdf.h`
(`PackedBsdf`), `reuse_constants.h` (128 bytes); `core/passes/restir_di.h`
(RTXDI's defaults, `BiasCorrection`, `Permutation`, `VisibilityReuse`, the
visibility limits); `core/frame/graph_file.h`, `schedule.h`, `history.h`;
`metal/reuse/unbiased.metal.h` (both modes), `reservoir.metal.h`,
`temporal.metal.h`, `spatial.metal.h`, `similar.metal.h`;
`metal/passes/restir_di/restir_di.metal.h` and the four passes' headers,
`bindings.h`; `metal/integrator/target.metal.h`, `path.metal.h`;
`metal/materials/bsdf.metal.h` (`pack_bsdf`, `unpack_bsdf`);
`metal/math/warp.metal.h` (`encode_unit`, `decode_unit`);
`metal/lights/sphere_light.metal.h`, `emitter.metal.h`;
`metal/sampler/sampler.metal.h`; `metal/frame/frame_images.h`,
`frame_resources.h`, `renderer.h`; `metal/acceleration/
scene_acceleration.h`.

Each compiled on its own (shaders and shared layouts with `xcrun metal
-std=metal4.0 -fmetal-math-mode=fast -Werror -I src`, shared layouts also
with `clang++ -std=c++20 -Wall -Wextra -Werror`, host headers with
metal-cpp), and the five existing kernels; every shared layout's size is
asserted on both sides and its trivial copyability on the host (P.5):
LightPoint 8, Reservoir 24, SurfaceRecord 32, PathState 36, PackedBsdf 16,
RestirDi 48, ReuseConstants 128.

### Findings and verdicts

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| W1 | I.24 | reuse/similar.metal.h | the first draft's similar(depth, record, depth, record): two (float, record) pairs that swap silently | fixed: similar(depth_here, normal_here, there, depth_there, settings), as before the rework; the material test, symmetric, is its own function |
| W2 | I.24 | reuse/temporal.metal.h, passes/restir_di/restir_di.metal.h | permuted_pixel(uint2 reprojected, uint2 permutation), and the permutation a bare uint2 in find_last_frame() | fixed: `BlockPermutation`, returned by frame_permutation() and taken by both |
| W3 | ES.3 | core/passes/restir_di.h | max_visibility_age and max_visibility_distance restated the reservoir's 255 and 127 | fixed: defined from contracts::tested_max_age and tested_max_offset |
| W4 | GPU.3 | passes/restir_di/restir_di.metal.h, restir_initial.h | said visibility reuse's query carries the unpacked surface (72 bytes); nothing after that query reads it | fixed: the streaming reservoir alone is alive across it; the shade computes the unshadowed light before its query and keeps the packed record, unpacking it again for the continuation, a trade the counters may reverse (GPU.10) |
| W5 | NL.17 | contracts/bsdf.h, frame/history.h, integrator/target.metal.h, sampler/sampler.metal.h | comment lines past the house's wrap | fixed: rewrapped (an existing member comment the rewrap touched restored as it was) |
| W6 | CDSA.21 | reuse/similar.metal.h | the material test's tolerances gate which reservoirs are reused; named, but constants, not injectable settings | kept, stated: in raytraced mode they decide noise only, and in basic mode they bound an already edge-confined bias; they become settings when a measurement varies them (Per.1, Per.2), as the geometric thresholds are |
| W7 | CDSA.31 | contracts/surface_record.h, bsdf.h | quantize, then rerank from fuller precision: the packed record serves the target and also the shade, with no full-precision rescoring | rejected, with reason: the target is a resampling weight, unbiased for any shape with the integrand's support; shading at full precision would put the 48-byte Bsdf, or a re-resolution with its texture reads, back in the shade's reads, and make the target and the integrand differ; production shades from its G-buffer. The packing's difference from the reference is stated where it is made and measured with raytraced mode's tolerance |
| W8 | GPU.2, CACHE.5 | contracts/reservoir.h | a 24-byte stride straddles a cache line for 1 in 8 gathered reservoirs (128-byte lines) | kept, stated: a quarter less traffic on every read and write, one gather a pixel at k = 1; supersedes R2 of the review of 70dbe59 |
| W9 | GDSA.18 | contracts/light_sample.h, bsdf.h | the rule is about block-scaled codes; the point, wo, normal and halves are fixed-size codes with no shared scale | cited for its decode-in-the-load-path part alone, which holds: codes are decoded where loaded and never stored decoded. R1's "GDSA.18 no longer applies" is reversed with the codes' return |
| W10 | GPU.5 | layout note (boiling filter) | RTXDI's boiling filter needs a threadgroup-wide reduction and barrier, the first in these kernels, and its result depends on the group's shape | recorded: the filter is a later, measured addition, named in the layout note with why, not dropped |
| W11 | Per.6 | passes/restir_di/restir_di.metal.h | milliseconds in a header | holds: each is arithmetic from the byte counts at a stated bandwidth, labelled so; the goal is stated as the implementation's to measure, and no measurement is in a header |

### Every category, rule by rule, over the changed headers

Notation as above: **holds**, a finding number, **convention**
(AGENTS.md "Guideline deviations"), or **n/a** with why. Rules this pass
does not name read as the first pass's verdicts, unchanged by the rework.

**C++ Core Guidelines.** In.0 n/a. **P**: P.1 holds (both modes by
numbered steps where they are stated); P.3 holds; P.4, P.5 hold (every
layout asserted; the new enums fail every switch that misses a case);
P.6 holds (settings, modes and media checked on the CPU, refused by name);
P.9 holds (the rework's purpose; W4); P.11 holds (one unit-vector map, one
pack and unpack, one reused_from()). **I**: I.1 holds; I.4 holds
(`BiasCorrection`, `Permutation`, `VisibilityReuse`, `BlockPermutation`
are types); I.5 holds (holds_light(), the media bound); I.7 holds
(unbiased.metal.h's postcondition, now per mode); I.9 holds as far as MSL
allows (find_last_frame()'s Frame stated); I.23 holds (shade()'s six
parameters each of a different type); I.24 W1, W2. **F**: F.1 holds
(holds_light, light_index, reused_from, encode_unit); F.8 holds (pack,
unpack, target pure); F.16 convention as before; F.21 holds (`Shaded`,
`Found`). **C**: C.2 holds (plain settings and layouts); C.1-C.183 n/a
otherwise (no classes added). **Enum**: Enum.3 holds; Enum.7, Enum.8
convention (a shared layout needs the underlying type, and zero is the
value a graph that does not read a setting holds). **R** n/a (no
resources added beyond ReuseImages' smaller arrays). **ES**: ES.3 W3,
else holds; ES.45 holds (every default, mask, shift and tolerance named);
ES.46 holds (every narrowing stated: halves, 16-bit codes, 8-bit offsets
and age, 24-bit count, 16-bit counter and media); ES.49 convention
(MSL's conversion syntax, `float(...)`); ES.101 holds (bit fields in
unsigned words); ES.102 holds (the signed offset limit). **Per**: Per.1,
Per.2 W6; Per.6 W11; Per.7 holds (every new choice a setting a
measurement varies); Per.16, Per.18, Per.19 hold (compact records,
contiguous own-pixel reads). **CP** n/a (no CPU threads). **E**: E.2
holds (the media bound refused by MetalError). **Con**: Con.4, Con.5 hold
(constants `constexpr`). **T**: T.10 as I.9. **CPL**, **A**, **NR**,
**GSL**, **Tools**, **FAQ**, **Discussion** n/a. **SF**: SF.8, SF.12
convention; SF.9 holds (restir_di.h now includes reservoir.h, which
includes nothing of the passes: no cycle); SF.10 holds (each header
includes what it names: restir_di.h reservoir.h, similar.metal.h bsdf.h
and warp.metal.h, target.metal.h bsdf.metal.h and warp.metal.h); SF.11
holds (compiled alone). **SL**: SL.con.4 holds (layouts trivially
copyable). **NL**: NL.17 W5; NL.8, NL.19 hold (`BlockPermutation` beside
the `Permutation` setting, in different namespaces, each named for what
it is); NL.2 holds.

**Performance.** **cache-layout**: CACHE.2 holds (the record from what
the target reads, as a G-buffer is); CACHE.4 holds; CACHE.5 W8 (every
field four bytes, no padding); CACHE.6 holds (record hot, path state
cold). **gpu**: GPU.1 holds; GPU.2 W8; GPU.3 W4; GPU.4 holds (basic
mode's shade ray is one query a group runs with the lanes that reuse a
test idle; raytraced as before); GPU.5 W10; GPU.6 holds (four
dispatches); GPU.7, GPU.8 hold (the shade's write-back is read by the
next frame behind the queue barrier, named in restir_shade.h); GPU.9
holds, F25 as before; GPU.10 holds. **gpu-dsa**: GDSA.3 holds (the frame's
permutation from the Sampler at pixel (0, 0), no CPU value); GDSA.6 holds
(recounted: 512 to 576 bytes a pixel); GDSA.15 F26 as before; GDSA.16
holds; GDSA.18 W9; GDSA.21 holds. **cpu-dsa**: CDSA.21 W6; CDSA.23 holds
(24 bytes of mergeable state); CDSA.31 W7; the rest n/a. **memory**:
MEM.4 holds (the shade writes this frame's half); MEM.9 holds (184 bytes
a pixel, allocated per size). **simd**: SIMD.5 holds (one gather a pixel
at k = 1). **copy-move**: COPY.6 holds. **lifetime**: LIFE.1 holds.
**codegen**, **concurrency**, **embedded**, **telemetry**, **wasm** n/a.

### Counts

| | Findings | Fixed | Kept, stated | Rejected, with reason | Recorded |
|---|---|---|---|---|---|
| C++ Core Guidelines | 5 (W1-W3, W5, W11) | 4 | 1 (W11 holds as written) | 0 | 0 |
| Performance | 6 (W4, W6-W10) | 1 (W4) | 3 (W6, W8, W9) | 1 (W7) | 1 (W10) |
| All | 11 | 5 | 4 | 1 | 1 |

New byte counts: LightPoint 8, Reservoir 24, SurfaceRecord 32 (PackedBsdf
16 of it), PathState 36, ReuseConstants 128; ReuseImages 184 bytes a
pixel. At RTXDI's defaults (M = 8, H = 20, k = 1, basic): 512 to 576 bytes
a pixel of intermediate traffic, at most 13 target evaluations (988
logical bytes of light data), 1 to 2 rays at the first rough surface; in
raytraced mode k + 2 = 3 rays.
