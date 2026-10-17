# The codebase against the two guideline corpora, rule by rule

*2026-10-16. Everything since Milestone 1, on branch `codebase-pass` from
60396c6: the platform-neutral core and the programs (area A), the Metal
backend's host code (area B) and the shaders (area C), each reviewed
read-only against both corpora, every category of each listed fresh and
walked rule by rule; then every finding decided here, fixed or judged not
valid with the reason, and the fixes made together (the user's rule: no
deferring a valid finding for "one intent"). The tests (area D) are the
next pass's; this one added only the tests its fixes need. Apple M3 Max,
the toolchain in `cmake/toolchain.json`.*

## Behavior unchanged

- **Byte for byte.** `serenity-headless`, release, 60396c6 against the
  pass, on the marbles at 480 × 270, frames 40 and 41: PNG of
  test_pattern, preview, path, path_mnee, restir, restir_ris,
  restir_spatial, restir_temporal and restir_pt; PFM of the seven that
  accumulate; restir_pt_features and restir_pt_denoised as PNG and PFM with
  `--features`; the features graph alone; restir_pt_denoised_small as PFM;
  restir_pt_denoised at 1920 × 1080, frames 120 and 121, PFM. 84 files, as
  Milestone 3's pass. `diff -r` finds none differ, after each group of
  commits and at the end.
- **Run to run, first.** Area C's P1 found that MNEE adds its requests'
  light by float atomics in arrival order, so its graphs need not repeat to
  the bit. Measured before anything else: 60396c6's build rendered the 84
  files twice, and the two runs were the same to the bit, MNEE's graphs
  among them. So the byte diff above is strict for every graph, not within
  a spread; the determinism level is now declared where the adds are
  (metal/passes/mnee/mnee.h, contract 16).
- **The gates.** All three presets build with 0 warnings; ctest green on
  each; `make check` green.

## The two defects

**C1: a replayed shift read a shape record 64 GiB past the records.** With
MNEE on (graphs/restir_pt.toml and restir_pt_denoised.toml, the demo's),
every replayed shift called `mnee::chain_after(chain, lobe,
records[here.surface.primitive].kind)` at its first vertex, where `here`
is the destination's primary hit, whose primitive is `no_primitive`
(0xffffffff). The kind was then unused (the chain is off there, and
`chain_after` returns before looking at it), so the images were right by
the order of a condition's operands. The baseline's IR showed the load
executed unconditionally, before the branch. Fixed: `chain_after(chain,
lobe, records, primitive)` reads the record only on a chain at a
refraction, the one case its kind decides; the IR now has the load behind
the branch (path_trace_mnee, restir_pt_temporal, restir_pt_spatial,
restir_pt_initial). Tested in tests/gpu/mnee_test.cpp under Metal's shader
validation: every case of `chain_after()`, the `no_primitive` ones among
them. What the test cannot show, measured: a probe kernel that read
`records[no_primitive]` through the argument table, with validation on and
off, and one that read an unmapped address, both read 0 and stopped
nothing. Neither shader validation nor the GPU sees such a read on this
machine, so the read's absence is the source's and the compiled kernels',
not a test's.

**C14: a full MNEE list reported its lost requests through a NaN.** Where
`mnee::enqueue()` found the list full, the path pass returned `float3(NAN)`
and ReSTIR PT's tree added it to `seen`, "so the frame counts it". The
shaders compile with fast math, which assumes no NaN; an `nnan` operation
on one is poison, and the lost caustic could have been neither added nor
counted. Fixed out of band: every writer adds to the list's count, so the
count passes the capacity by exactly the requests that found no room; the
MNEE pass's one-thread grid kernel adds that excess to a counter of its
own. The non-finite counter's class is now `FailureCount`
(metal/film/failure_count.h), one per kind of failure, and the renderer
keeps two: `non_finite_samples()` and `mnee_lost_requests()`. The window's
title shows a lost request; the headless renderer fails the run on one,
as it does on a sample not finite. The path's sample keeps its other
light, finite. Tested: a light in a glass sphere in a closed white
cavity, where every bounce off the wall writes a request and a path
writes several; the list (one a pixel) fills, the lost count is above 0
for the path pass with MNEE and for ReSTIR PT with it, no sample is not
finite, and the image is finite.

## The tests the fixes added

Only what the fixes need; area D's review of the tests themselves is the
next pass's.

- tests/gpu/mnee_test.cpp: `chain_after()`'s every case, `no_primitive`
  among them (C1); a light in glass in a cavity filling the list (C14).
- tests/gpu/kernels/mnee_probe.{h,metal}: `ChainQuery` and the
  `mnee_chain_after` probe.
- tests/graph_file_test.cpp: `mnee = false` refused where no pass reads it
  (F1). tests/denoise_core_test.cpp: a repeated JSON member (D3), an
  unreadable export (D2), the messages F8 and D1 changed.
  tests/schedule_test.cpp: the tone map's messages (F8).
  tests/options_test.cpp: options given twice (B1). tests/flight_test.cpp:
  `job()`, `reason()` (N5). tests/test_check_toolchain.sh: a missing and a
  malformed pin file (T1).
- Renamed for the fixes: frame_errors_test.cpp's `NonFinite` is
  `FailureCount`; path_test.cpp's reference to its header.

## Coverage, by area and corpus

The reviewers' tables, summarized; their reports are in the session's
scratchpad, not the repository.

| Area | Core Guidelines: categories listed / walked with findings | Findings | Performance: categories listed / with findings | Findings |
|---|---|---|---|---|
| A: core, app, headless, measure, tools | 23 listed; findings in P, I, F, C, ES, CP, E, Con, SF, SL, NL; walked clean: In, Enum, R, Per, T, CPL, A, NR (GSL, Tools, FAQ, Discussion n/a) | 41, and 1 in tools (no corpus) | 13 listed, all walked; none (per-frame CPU work tens of microseconds against a 25-150 ms frame) | 0 |
| B: Metal host | 23 listed; findings in P, I, F, C, Enum, ES, E, Con, SF, SL, NL; clean: R, Per, CP, T, CPL, A, NR | 32 (G1-G32) | 13 listed; GPU.8 | 1 (P1) |
| C: shaders | 23 listed; findings in P, I, F, ES, E, Con, SF, SL, NL; clean: C, Enum, R, Per, CP, T, A, NR | 22 (C1-C22) | 13 listed; gpu (GPU.3), gpu-dsa (GDSA.2) | 3 (P1-P3) |

Not raised by any reviewer: AGENTS.md's recorded deviations, what the
Milestone 3 pass judged not valid (its G22-G30), and the 2026-10-10
sweep's rejections.

## Area A: the core, the programs, tools

Lines are 60396c6's. The core's findings outside the contracts were fixed
in their own worktree and merged (dc9bf33); the contracts' (C1-C5) here,
since both sides read them.

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| C1 | ES.45, P.5, ES.3 | contracts/denoise.h | `input_channels`, `denoise_scores`, `head_outputs` hand-computed | fixed: the two derived from their sources; input_channels' padding asserted |
| C2 | P.5, SL.con.4 | contracts/denoise.h, path_reuse_settings.h | MedianState without its trivially-copyable assertion; contract 15's two host blocks split and reversed | fixed |
| C3 | ES.3, P.5 | contracts/path_reuse_settings.h | `pairing_starts`, `pairing_entries` restate `pairing_widths` | fixed: derived in place (the Metal compiler takes the constexpr reads) |
| C4 | NL.8 | camera.h, gradient_sky.h, box.h, tone_map.h, reuse_settings.h, restir_di.h, restir_pt.h | padding named four ways; `bloom_levels` an `unsigned int`; `u` suffixes uneven | fixed: interleaved paddings `padding0/1`, trailing ones `padding[N]`; `uint32_t` and `u` throughout |
| C5 | ES.3, ES.45 | linear_image.cpp, pfm.cpp, headless/main.cpp | contract 13's 3 and 4 channels named privately thrice | fixed: `linear_image_channels`, `accumulated_image_channels` in contract 13 |
| F1 | E.2, I.5 | graph_file.cpp | `mnee = false` accepted in a graph no pass of which reads it | fixed: refused at its line whatever its value; tested |
| F2 | NL.2, I.1 | graph_file.h | the format omits `[restir_pt]` and `mnee` | fixed |
| F3 | ES.3, F.3 | graph_file.cpp | two readers' four lambdas line for line; `read_schedule` 95 lines | fixed: one `SettingsReader<Settings>`; `read_schedule` in parts; messages byte-identical |
| F4 | NL.2 | schedule.cpp | two orphaned comment blocks | fixed |
| F5 | ES.3, F.10 | schedule.cpp | `restir_broken`, `restir_pt_broken` repeat one rule | fixed: `family_broken()` |
| F6 | F.3, ES.12, ES.3 | schedule.cpp | `invalid(RestirPt)` 67 lines; a lambda hiding `setting()` | fixed: four parts; `setting()`, `unread()` take the family; messages byte-identical |
| F7 | NL.3 | schedule.h | stale comments | fixed |
| F8 | ES.3 | schedule.cpp, network.cpp | floats printed by `std::to_string` beside `text()` | fixed; the tests matching those messages updated |
| F9 | ES.3 | history.cpp | two joins repeat one start-over branch | fixed: `HeldFrames` |
| F10 | ES.3 | eight places | "W x H" from two `to_string`s | fixed: `frame::to_string(Extent)`, and in the backend's four such messages |
| N1 | ES.3, F.10 | flight.cpp | Bezier controls computed thrice | fixed; flight_test's pinned loop passes, bit for bit |
| N2 | ES.3 | flight.cpp | `close_loop`'s inner loop is `build()` again | fixed; bit for bit |
| N3 | ES.3, F.10 | flight.cpp, wander.cpp | outward rounding twice in one family | fixed: `core/animation/rounding.h` |
| N4 | ES.3 | flight.cpp | the Float3 narrowing inline twice | fixed |
| N5 | E.16, C.9 | flight.h | `FlightsError`'s copy could throw; public members | fixed: a `std::runtime_error` member, `job()`, `reason()`, asserted nothrow-copyable |
| N6 | P.5, ES.45 | flight.h, flight.cpp | `Segment::numbers` a bare 19, its tables unchecked | fixed: `segment_numbers`, every table asserted to fit |
| N7 | CP.9 | make_batch's workers | never run under ThreadSanitizer | fixed by trying: TSan's runtime crashes before main on this macOS (a nine-line `std::thread` program too, Apple clang 17 and Homebrew clang 21; `EXC_BAD_ACCESS` in `__tsan::SlotLock`); recorded in 2026-10-10-sanitizers.md; no preset, since one that cannot start would claim a check that never runs |
| N8 | ES.45 | wander.h, wander.cpp | the term arrays' literal 3 | fixed: `wander_terms` |
| N9 | Con.3 | animate.cpp | `check_movers` takes a mutable span | fixed |
| S1 | F.3 | scene.cpp | `make_animation` 85 lines | fixed: gather, fly, record |
| S2 | ES.3, T.1 | scene.cpp | two `words()` with one body | fixed: one template |
| S3 | NL.2 | scene.cpp | a comment names no `range` | fixed |
| S4 | C.12, C.21 | scene.cpp | `ShapeReading` copyable, holding references | fixed: copy and move deleted |
| S5 | I.24, ES.3 | swarm.cpp | `start_of` takes one axis twice | fixed |
| H1 | ES.3 | shapes/primitive.h | `Bounds` the same two corners as contract 11's `Box` | fixed: `using Bounds = contracts::Box`; Box's comment names both uses |
| K1 | ES.20 | thin_lens.h | `ViewAxes` members without initializers | fixed |
| D1 | ES.45, ES.3 | network.h, network.cpp | the U-Net's four levels and "1, 2 and 4" as literals | fixed: from `unet_levels`, `spacing_pixels` |
| D2 | I.10, SL.io.2 | network.cpp | `read_file` could not see a read error | fixed: a sized read, `gcount()` compared, "cannot read"; tested (on macOS a directory in a file's place reads 0 bytes) |
| D3 | SL.io.2 | json.cpp | a repeated member name accepted | fixed: refused with its byte; tested |
| D4 | SF.10 | json.h | `std::nullptr_t` without `<cstddef>` | fixed |
| P1 | ES.45 | pairing.h, pairing.cpp | 254, "0.8", 24 and the seed step as literals | fixed: `largest_pairing_width` asserted against every width, messages from the constants, `texture_seed_step` |
| X1 | SL.str.2 | packed_range.cpp, reference.cpp | `const char*` names | fixed: `std::string_view` |
| X2 | NL.4 | nine lines, continuations, a double blank | past 120 columns | fixed |
| B1 | E.2, SL.io.2 | app/options.cpp, headless/options.cpp | an option given twice keeps its last value | fixed: refused by name; tested |
| B2 | E.14 | app/main.cpp | a bare `runtime_error` for "shows no image" | fixed: `app::OptionsError` |
| B3 | F.3 | headless/main.cpp | `main` 170 lines, four lambdas | fixed: `check_graph()` and a `Run` class |
| T1 | (tools) | check_toolchain.py | a missing or malformed pin file is a traceback | fixed: a named failure, status 1; tested |

None judged not valid.

## Area B: the Metal host

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| G1 | I.1, P.3 | scene_acceleration.h | `Keep`, a constructor, `update_kept()`, `traced()` declared and documented, defined nowhere; their members, `last_acceleration`, binding 13 | fixed: removed, `structures_` sized `frames_in_flight`, ReSTIR DI's table renumbered (last_glows 13, last_transforms 14; 15 buffers); the design for keeping last frame's structure is a QUEUE item the renderer's refusal points to; the refusal stays, tested (restir_test.cpp) |
| G2 | F.3 | renderer.cpp | the constructor and `record()` 275 lines each | fixed: `make_steps`, `make_images`, `make_restir_di`, `make_restir_pt`, `make_mnee`, `make_scene`; `take_prepared`, `frame_resources`, `animate`, `denoise_resources`, `record_steps` |
| G3 | C.1, R.5 | renderer.h | forty members, each family's a run of parallel unique_ptrs | fixed: `RestirDiFamily`, `RestirPtFamily`, `MneeFamily`, `FeatureFamily`, `DenoiseFamily`, each held in place in a `std::optional`, each binding its share |
| G4 | ES.3 | renderer.cpp | "must be prepared" written twice, incompletely | fixed: `prepares()` |
| G5 | ES.3 | renderer.cpp | the float-range time test twice | fixed: one `check_time()` |
| G6 | ES.3 | renderer.cpp | the media and packed-BSDF checks twice | fixed: `check_packable()` |
| G7 | ES.45 | renderer.cpp | `dec[3]` | fixed: `dec.back()` |
| G8 | ES.40, P.5 | renderer.cpp | a four-deep nested ternary for the table's size | fixed: `buffer_bindings()`, the most of the graph's families, each family's count asserted against its bindings; the other two ternaries if/else |
| G9 | NL.2, NL.3 | renderer.cpp, renderer.h | stale binding counts; two asserts missing | fixed: the texts cite the asserts; `path::mnee_header`, `features::motion` asserted |
| G10 | ES.3, SF.3 | six owners, three more by hand | a resident allocation written nine ways | fixed: `ResidentBuffer`, `ResidentTexture`, `make_resident_buffer()`, `make_resident_texture()` (metal/device/resident.h) |
| G11 | ES.3, F.10 | eight places | the dispatch barrier spelled out | fixed: `barrier_after_dispatch()` |
| G12 | ES.3, P.5 | eleven passes | the scene's bindings and their test per pass | fixed: `bindings::scene_view`, every pass's indices asserted equal to it, `bind_scene()` |
| G13 | ES.45, P.5 | restir_pt_temporal.cpp, restir_pt_spatial.cpp, mnee.cpp | the replay and solve groups the host's own 64s | fixed: `contracts::replay_group`, `contracts::mnee_solve_group`, read by both sides |
| G14 | ES.45 | restir_pt_initial.cpp | the light tiles' 256 | fixed: `tiles_group`, the host's (the kernel reads only its grid position) |
| G15 | ES.45, I.4 | mnee_list.cpp, frame_images | the header's words by index; `offset * 2` | fixed: `mnee_count_word`, `mnee_capacity_word`, `mnee_header_bytes`; `counter_bytes` |
| G16 | NL.2, P.6 | support.h, scene_block.h | "256 is what Metal asks of a constant buffer's offset", with four placements not on it | fixed: Apple's Metal Feature Set Tables (May 21, 2026 edition, page 7, "Resources": "Minimum constant buffer offset alignment", 4 B for Apple2 to Apple10) named once, `constant_offset_alignment`; the camera's 128, the dispatch parameters' 64 and both grids' 64 asserted against it and their types' alignment; 256 stays as the backend's chosen start, and the comments say it is a choice |
| G17 | E.4 | frame_images.cpp, denoise_images.cpp | histories joined before the arrays exist | fixed: joined on a copy, kept once the arrays are made |
| G18 | P.7, I.5 | presenter.cpp | `resize()` unchecked | fixed: `check_texture_size` |
| G19 | ES.3, I.5 | renderer.cpp, feature_images.cpp | one readback twice, its precondition unchecked | fixed: `read_back()` (metal/device/readback.h), a queue barrier before its copies, so it waits for any frame in flight |
| G20 | I.6, P.6 | frame_array.h | `view<T>` truncates silently | fixed: throws on a partial T |
| G21 | NL.2, NL.4 | several | reflows, stranded asserts, submission.h's order | fixed: every line to 120 columns; `frames_in_flight`, `Completed`, `FrameSlot` first in submission.h, the description above `Submission` |
| G22 | I.24 | denoise_images.cpp | offset parameters always 0 | fixed: dropped |
| G23 | ES.12, NL.19 | denoise_images.cpp, frame_images.cpp, denoise.cpp | names hiding functions | fixed: `make_conv`, `make_pool`, `param_stride`, `h`, `copy`, `max_pool` |
| G24 | SL.str.2 | device.h, support.h, frame_images.h, denoise_images.h, mnee_list.h | `const char* what` | fixed: `std::string_view`; `Library::compute_pipeline` keeps `const char*` (NS::String needs a terminated string) |
| G25 | Con.2, Con.3 | non_finite.h | a const member handing out a writable pointer | fixed: a const and a non-const `counter()` |
| G26 | SF.10 | renderer.h, renderer.cpp, tone_map.cpp | names through other headers | fixed |
| G27 | NL.2 | path.h | "the naive path tracer" | fixed: "the path tracer without reuse, next event estimation weighed against BSDF sampling by MIS" (and in path.metal.h) |
| G28 | ES.45 | denoise.cpp | `enc[0..3]` by hand, weights at `2 * i` | fixed: loops over `DenoiseImages::levels`; `weights_index()`, `bias_index()`, `head_index` |
| G29 | Enum.3 | bindings.h | plain enumerations not among AGENTS.md's deviations | fixed: recorded there |
| G30 | F.6 | library.h, scene_buffers.h, offscreen.h | getters that cannot throw, not noexcept | fixed |
| G31 | I.5 | three PT pass headers | `record()`'s throw unstated | fixed |
| G32 | ES.3 | restir_pt_temporal.cpp, restir_pt_spatial.cpp | one four-dispatch sequence twice | fixed: `ShiftReplay` (metal/passes/restir_pt/shift_replay.h) |
| P1 | GPU.8 | renderer.cpp | a barrier between restir_pt_shade and restir_pt_duplication that orders nothing | not taken, by its numbers: `frame::reads_pass_before()` (false for the duplication pass alone) and the renderer recording the barrier only where it is true were built and timed against the pass without them, interleaved: no gain (restir_pt at 1920 × 1080 39.13 and 40.00 against 39.15 and 40.38 ms; restir_pt_denoised 76.61 and 78.73 against 76.59 and 78.25 at the window's size, 23.80 and 23.51 against 23.85 and 23.54 at 1080p). The shade is one dispatch the duplication rarely overlaps; the barrier stays, its hazard-free status recorded here. The analysis holds (area C confirmed the shade binds neither array) |

## Area C: the shaders

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| C1 | ES.65, SL.con.3, I.5 | shift.metal.h | the shape record read at no_primitive | fixed (above) |
| C2 | ES.3, P.1 | path.metal, restir_initial.metal, features.metal, restir_pt.metal.h | the camera's ray drawn four times | fixed for three: `pixel_ray()` in camera/thin_lens.metal.h, used by the features pass, ReSTIR DI's initial pass and ReSTIR PT's `primary_ray()`. Not taken for the path pass: through `pixel_ray()` its kernel's prologue compiled otherwise (its loop the same) and the path graph measured 2% slower at 3456 × 2234 (30.62, 30.61, 30.95, 30.95 against 30.10, 30.11, 30.10, 30.43 ms, ABBA). It writes the draw out again, with a comment naming `pixel_ray()`; its compiled `path_trace` is the baseline's, function for function. Byte identity holds the four equal |
| C3 | ES.3 | restir_pt.metal.h | the temporal search's tries, ring and start written again | fixed: from metal/reuse/temporal.metal.h, with `ring_neighbor()`, `draw_ring_start()` |
| C4 | ES.45, ES.3 | the two PT reuse kernels | the replay items' radix 4 | fixed: `replay_slots`, `replay_item()`, `replay_pixel()`, `replay_slot()`; both reuses' slots asserted to fit |
| C5 | ES.45, Con.5 | duplication.metal.h | `others = 288.0f` | fixed: from the window |
| C6 | ES.3 | path.metal.h | MNEE's step 5 a copy of the path tracer's | not taken, by its numbers: built (`draw_next_event()`, `unshadowed_term()`, Manifold's hooks Radiance's) and byte-identical, but it changed both path kernels' compiled `next_event`, and path_mnee measured +0.3 ms (0.5%) at 3456 × 2234 (58.28, 58.15 against 58.04, 57.76 ms, interleaved; the whole pass against 60396c6, ABBA: 58.38, 58.38, 58.41, 58.38 against 58.30, 58.02, 58.05, 58.03). Reverted (821e24c): the copy stays, as the path loop's copies did in 2026-10-11-path-split-cost.md |
| C7 | ES.3, F.10 | the PT kernels, mnee.metal | the SIMD append and the indirect grid written twice and thrice | fixed: `simd_reserve()`, `indirect_grid()` (metal/device/append.metal.h) and one `IndirectGrid` both sides read (metal/device/indirect_grid.h) |
| C8 | ES.3, ES.45 | reservoir, visibility, pairing | a signed-byte decode thrice; the golden ratio unnamed | fixed: `signed_byte()`, `tested_offset()`, `golden_ratio_32` |
| C9 | ES.3, ES.45 | similar.metal.h, target.metal.h | the kind and Rec. 709 written again | fixed: `packed_kind()` in contract 2, `rec709_luminance` in math/srgb.metal.h, each use keeping its expression's form |
| C10 | ES.45, ES.3 | shift.metal.h, reconnection.metal.h | unnamed floors | fixed, step 1: `least_cosine`, `least_geometry`. Step 2, the inline footprints written as `footprint()`, not taken: it changes the bits (`d^2 / (p cos)` against `1 / (p cos / d^2)`), and the names already say what is computed |
| C11 | ES.45 | restir_di.metal.h | the ring's and permutation's sizes as literals | fixed: `ring_size`, `permutation_range` |
| C12 | ES.45, ES.3, P.5 | mnee.metal.h, shape_point.metal.h | the request's shapes packed by hand; a box point's 16-bit codes | fixed: `mnee_shape_bits`, `mnee_shapes_a_word` in contract 16, the packing a loop as the unpacking is; shape_point's codes by warp.metal.h's names |
| C13 | SF.10 | four shader files | `mnee::` and `transform_*` through other includes | fixed |
| C14 | I.10, E.27, P.7; GEN.8 | path.metal.h, restir_pt.metal.h | the full list's loss through a NaN | fixed (above) |
| C15 | NL.2 | path.metal.h | `Stopped::walk`'s comment false half the time | fixed |
| C16 | F.3 | shift.metal.h, mnee.metal.h, restir_pt.metal.h | three long functions | fixed, all three kept: `reconnect()` into `reconnect_to_light()` and `reconnect_to_surface()`; `manifold_light()`'s steps 3 and 4 into `chain_transmittance()` and `chain_visible()`; the tree's NEE's RIS draw into `resample_lights()` (its offer, the rest of `next_event()`, kept whole: what is left is one step). Each forced inline, byte-identical; their kernels' IR changed, so timed: restir_pt at 1080p and restir_pt_denoised at both sizes unchanged against 60396c6 (below), path_mnee's solve unchanged against the build before it (58.15 against 58.15 ms in the undisturbed round) |
| C17 | P.1, P.5, ES.45 | three files | the streams' purposes kept apart by comments | fixed: one table in sampler.metal.h, each span's bound asserted where it is drawn |
| C18 | I.1, ES.3 | restir_pt.metal.h, uniform_light.metal.h | the uniform probability written in the Pass and thrice in the axis | fixed: `selection_probability()`; `select_light()` built on `select_light_index()`, which the file had kept apart for fear of moving the path tracer's image (it does not move) |
| C19 | ES.3 | restir_di.metal.h, restir_pt.metal.h | last frame's position written by hand | fixed: `transform_to_world()` |
| C20 | NL.19, ES.45 | restir_di.metal.h | the medium masked with the counter's mask | fixed: `path_medium_mask` |
| C21 | ES.12 | restir_pt.metal.h | a local `emitted` in `emitted()` | fixed: `glow` |
| C22 | NL.17 | nine lines | past 120 columns | fixed |
| P1 | GDSA.2 | mnee.metal | an undeclared non-deterministic reduction | fixed: declared in mnee.h and contract 16, and measured reproducible (above); a fixed-order fold not taken, nothing needing it |
| P2 | GPU.3, Per.6 | mnee.metal.h | the solve recomputes each crossing's frame | not taken, by its numbers: each crossing's Vertex found once a chain and passed to `constraint_blocks()` and the step (byte-identical, the same function of the same inputs) made path_mnee 67.49 and 67.20 ms against 58.39 and 58.17 at 3456 × 2234, and restir_pt_denoised 81.16 and 77.87 against 78.25 and 76.59: the four Vertices, 84 floats, spill, as area C feared. Step 2 (s and t from the Vertex) was not tried: not byte-identical, and it would not undo the spill |
| P3 | GPU.3; NL.2 | mnee.metal.h | the registers claim, against seed_segment's insertion | fixed by the claim: narrowed to the solve's loops. The insertion's register use cannot be read on this toolchain without the GPU tools, so it is left as it is, said so |

## Timings

GPU time a frame, median of 179 frames after 21 of warm-up, two frames in
flight, the marbles from t = 120 advancing a sixtieth of a second a frame;
the host's median time to prepare and record a frame beside it. The
harness is outside the repository (serenity-scratch's, extended to take
the size and graph from the environment). The GPU idle before each run
(utilization at most 5% for 6 s), no serenity window. Builds: 60396c6
and each group of this pass's commits, built apart in the scratchpad.

**What disturbed the runs, said first.** The machine ran on battery
throughout (85% falling to 47%). ReSTIR PT at 3456 × 2234 runs 30 s at
full load, and after one or two such runs in a row the GPU throttled:
the same build measured 148, 170, 204, 195 ms in four consecutive runs,
the idle check passing between them. Those runs are dropped; restir_pt at
the window's size is compared only through restir_pt_denoised, whose light
is that size's quarter, and at 1080p. A first protocol, seven builds in a
row per case, also showed an order effect of up to 1 ms that fell with
position; the decisions below rest on ABBA runs (A, B, B, A, twice, after
a warm-up run of A that is not counted), each cell every kept run.

**The pass against 60396c6** (ABBA, 60396c6 against dc7ce99, the pass
with C2 at all four sites and C6, before both were undone for the path
pass):

| | 60396c6 | the pass | |
|---|---|---|---|
| path, 3456 × 2234 | 30.45, 30.39, 30.18, 30.12 | 30.56, 30.57, 30.18, 30.55 | +0.18 ms: C2 and C6 (below), both since undone for these kernels |
| path, 1920 × 1080 | 8.08, 8.09, 8.05, 8.08 | 8.10, 8.10, 8.09, 8.14 | +0.03 ms, the same |
| path_mnee, 3456 × 2234 | 58.30, 58.02, 58.05, 58.03 | 58.38, 58.38, 58.41, 58.38 | +0.29 ms: C6, since undone |
| restir_pt, 1920 × 1080 | 41.72, 40.30, 39.90, 39.27 | 41.31, 40.84, 39.68, 39.30 | the same (40.30 against 40.28 on average; the drift is the GPU's, both alike) |
| restir_pt_denoised, 3456 × 2234 | 76.92, 76.93, 77.10, 77.78 | 76.94, 76.92, 77.26, 77.56 | the same |
| restir_pt_denoised, 1920 × 1080 | 23.58, 23.52, 23.50, 23.53 | 23.54, 23.56, 23.54, 23.57 | the same |
| host, record a frame (restir_pt_denoised, 1080p) | 160.7, 150.5, 151.1, 152.0 us | 166.9, 154.5, 156.7, 163.3 us | +7 us on average, within the runs' spread, against a 23.5 ms frame: the renderer's restructuring (G2, G3) and its helpers cost nothing that shows |

**What was undone, and why, measured:**

| Change | Against | Graph | Without (ms) | With (ms) | Verdict |
|---|---|---|---|---|---|
| C2 at the path pass | 60396c6 (ABBA) | path, 3456 × 2234 | 30.10, 30.11, 30.10, 30.43 | 30.62, 30.61, 30.95, 30.95 | +0.6 ms, 2%: written out again; its `path_trace` now the baseline's, function for function |
| C6 | the build before it | path_mnee, 3456 × 2234 | 58.04, 57.76 | 58.28, 58.15 | +0.3 ms: reverted |
| P2 (step 1) | the pass | path_mnee, 3456 × 2234 | 58.39, 58.17 | 67.49, 67.20 | +9 ms: not taken |
| area B's P1 | the pass | restir_pt, 1080p; restir_pt_denoised | 39.15, 40.38; 78.25, 76.59 | 39.13, 40.00; 78.73, 76.61 | no gain: not taken |

What is committed, then, has the baseline's compiled path kernel (the
path graph unchanged by construction: `path_trace`'s IR is 60396c6's,
function for function), path_mnee's MNEE functions changed only by the
defect fixes C1 and C14 and the named packing (C12) and append (C7),
measured equal to 60396c6 before C6 (58.04 and 57.76 against 58.05 and
57.81 ms), and ReSTIR PT's and the denoiser's graphs as above.
Milestone 3's figures stand: the window's frame 76.9 ms, 1080p 23.5.

## Counts

| Area | Corpus | Findings | Fixed | Not valid | Not taken, by its numbers |
|---|---|---|---|---|---|
| A | Core Guidelines | 41 | 41 | 0 | 0 |
| A | tools (no corpus) | 1 | 1 | 0 | 0 |
| A | Performance | 0 | | | |
| B | Core Guidelines | 32 (G1-G32) | 32 | 0 | 0 |
| B | Performance | 1 (P1) | 0 | 0 | 1 (P1: no gain) |
| C | Core Guidelines | 22 (C1-C22) | 21 (C2 at three of its four sites; C10's first step) | 0 | 1 (C6: +0.5%), and C2 at the path pass (+2%) |
| C | Performance | 3 (P1-P3) | 2 (P1 declared, P3's claim) | 0 | 1 (P2: +16%) |

Two of them defects (C1, C14), fixed with their tests. Every graph the
same to the bit as 60396c6's, all three presets building with no warning
and testing green, `make check` green.
