# Verdicts: src/core (branch `worktree-agent-a7c205fa38ca68978`, from marbles c0bc7d1)

Commits: `7755d04` (AGENTS.md "Guideline deviations"), `25f3b70` (every code,
test and research-note fix below). Line numbers are the reports' (pre-fix).

Tests: both presets build with -Werror; `serenity_tests` 108 cases / 1595
assertions pass (was 92 / 1522) in release and debug;
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 serenity_gpu_tests` 75 cases pass
(Apple M3 Max); `tools/check_boundaries.sh` and its self-test pass. Every
scene's animation (each moving transform and glow at 3000 times, each
loop and flash start) is bit-identical before and after, all four scenes.

Measurements (M3 Max, release flags, machine shared with other builds, load
avg 5 to 36; ratios within a round are the reliable figure):
- `scene::load("scenes/marbles.toml")` (616 fireflies, 51 still shapes),
  min / median of 9, three interleaved rounds: before 428/459, 461/475,
  552/587 ms; after the CDSA.32 change 110/124, 155/165, 151/160 ms (3-4x).
  The sweep's other changes alone ("mid"): 451/461, 513/545, 567/583 ms
  (no change). brass_sphere_flight: 1.8 -> 1.3 ms.
- Load profile before (macOS `sample`, 39946 samples): distance query 92.2%
  (shapes::distance 51%, placement() 36%, scan 5%); allocation, strings and
  TOML 25 samples, 0.06%. After: StillShapes::distance 75%, closed forms and
  libm 20%.
- Per frame, `animate()` for 616 movers + 616 glowers: median 43 us (34 min)
  quiet, 66-76 us loaded; unchanged by the sweep; under 0.5% of 16.7 ms.

## s1-core.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s1-core | animation/flight.cpp:141 | ES.46 | fix | count compared as a double before the int64 cast; bound named `flight_most_steps`, injectable as `FlightParams::most_steps`; tests: speed 1e30 and most_steps 2 are Refusals | 25f3b70 |
| s1-core | animation/glow.cpp:30 | ES.46 | fix | `least_period` (1e-3 s) enforced by the reader; `glow()` refuses \|k\| >= 2^62 or NaN with invalid_argument before converting; tests for both | 25f3b70 |
| s1-core | camera/thin_lens.cpp:48-53 | I.6/I.7 | fix | framing and its checks in double, every test in pass-only form; 3e38 and 1e20 cameras frame finite; shader_form checks its preconditions; tests | 25f3b70 |
| s1-core | scene.cpp:933-937, graph_file.cpp:155-159 | I.10, SL.io.2 | fix | file sized by filesystem::file_size, read whole, gcount checked; a directory is refused; tests in scene_test and graph_file_test | 25f3b70 |
| s1-core | flight.h:202-204, flight.cpp:545 | E.17 citation | fix | E.17 dropped; the exception transport (catch in worker, exception_ptr, rethrow in caller) described without an ID | 25f3b70 |
| s1-core | schedule.h:107 vs schedule.cpp:10 | ES.45 | fix | static_asserts hold `kinds` to the enum's length and last value; claim reworded; test that every kind round-trips by name | 25f3b70 |
| s1-core | scene.cpp:503 | C.12 | fix | Obstacles deletes copy/move (C.67), so StillShapes (now StillObstacles, holding its own placed copy) is not copyable | 25f3b70 |
| s1-core | flight.cpp:163-197 | Enum.3 | fix | `enum class Purpose : std::uint64_t`, values unchanged (the keying), converted in Draws | 25f3b70 |
| s1-core | scene.cpp:487, graph_file.cpp:38, scene.h:291 | SF.10 | fix | `<algorithm>`, `<initializer_list>`, `<cstdint>` included directly | 25f3b70 |
| s1-core | wander.cpp:45-47 | E.2/I.10 | fix | message names body's condition; test checks it | 25f3b70 |

## s2-core-cg.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s2-cg | shapes.cpp:146,158; flight.cpp:124 | P.6 | fix | std::logic_error past every switch (no_case); tests with out-of-range kinds for distance/touches/object_bounds and a corrupt segment | 25f3b70 |
| s2-cg | shapes.cpp:134; motion.cpp:16,26; glow.cpp:68; flight.cpp:91; schedule.cpp x6 | P.6 | fix | same; tests for motion position/extent, glow, and all six PassKind functions | 25f3b70 |
| s2-cg | flight.cpp:527-536 | I.6 (SL.con.3) | fix | position() refuses a flight with no segments or loop <= 0 (invalid_argument); test with a default Flight | 25f3b70 |
| s2-cg | thin_lens.cpp:67; coated.cpp:26; frame_times.cpp:7; history.cpp:55,62; glow.cpp:39-48 | I.6 | fix | each precondition checked and thrown: empty size/invalid camera, ior <= 1 or NaN, end < start, frames out of order, s >= samples, loop <= 0; a test for each | 25f3b70 |
| s2-cg | animate.h:42; wander.h:78 | I.6 citation | fix | cite I.5 and E.2 | 25f3b70 |
| s2-cg | scene.cpp:663-664 (flight.cpp:587) | I.4 | fix | FlightsError carries `reason`; reader uses it; test checks job, reason and what() | 25f3b70 |
| s2-cg | frame_times.h:55 | I.4 | fix | `frame::Seconds total_` | 25f3b70 |
| s2-cg | graph_file.h:50; scene.h:314 | I.24 | reject | parse(text, source): a swap parses the source name as TOML and fails at once with an Error (no camera / no passes), never silently; I.24's risk is a silent swap; a strong type would touch every test call site for no caught bug | — |
| s2-cg | history.h:121,124 | I.24 | cross-area | plan_headless(3 bools) and sample(frame, s) want a parameter struct; called from src/headless/main.cpp, so the signature change is the headless owner's (listed below) | — |
| s2-cg | wander.h:79; flight.cpp:211,:333 | I.24 | fix | WanderParams (named fields); nearby(..., reach, Room{below, above, around}); hermite(params, End from, End to) | 25f3b70 |
| s2-cg | scene.cpp:712,598,775,509; flight.cpp:211; wander.h:79 | I.23 | fix | ShapeReading class holds the reader's state (read_swarms 9 -> 2 params, make_animation 7 -> 1, read_shapes 7 -> 1); nearby 7 -> 5; make_wander 6 -> 3 | 25f3b70 |
| s2-cg | scene.cpp:200 | I.22 | fix | world_words() built from world_extent when an error is raised | 25f3b70 |
| s2-cg | scene.cpp:433,669 | I.12 | fix | SwarmEntry::table is std::reference_wrapper<const toml::table> | 25f3b70 |
| s2-cg | thin_lens.h:24; scene.cpp:334,775,712 | F.20 | fix | camera::invalid() returns optional<string_view>; read_albedo returns Albedo; read_shapes/read_swarms write ShapeReading's members | 25f3b70 |
| s2-cg | flight.cpp:373-525; scene.cpp 598-707, 323-428, 775-868, 203-278; schedule.cpp:111-191 | F.3 | fix | make_flight split per step (draw_episodes, close_loop, lay_out, schedule_flashes, check_numbers); scene per kind (read_checker/wood/swirl/absorbing/coated/...; read_sphere/read_box; make_wander/make_glow/report); invalid() per rule | 25f3b70 |
| s2-cg | draw.h:18,25; transform.h:51,59,65; sphere.h:22; box.h:33; bsdf.h:126; animate.h:71,75; flight.cpp:37-45; wander.cpp:17 | F.4 | fix | constexpr on draw.h, transform helpers, sphere/box bounds, aims_at_lights (compiles in MSL), flight's Vec3 ops; wander's splitmix64 removed (shared); animate.h's moves/changes read a run-time vector, so noexcept only | 25f3b70 |
| s2-cg | same set, thin_lens.cpp:12-34, shapes.cpp:29-43, coated.cpp:13, history.cpp:55,62 | F.6 | fix | noexcept on the non-throwing helpers (thin_lens, shapes' per-kind tests and rounding, fresnel, Vec3, draw, transform); history's two now throw by design (I.6 row), so not noexcept | 25f3b70 |
| s2-cg | wander.cpp:57-64,126-133; shapes.cpp:111-118 | F.10 | fix (in part) | one round_down/round_up (and outward()) in wander.cpp; shapes.cpp names its own pair. Sharing them across Animation and Shape rejected: families depend on contracts, not each other (file-mapping.md); a shared helper module is an architecture change | 25f3b70 |
| s2-cg | shapes.h:38 | F.16 | fix | object_bounds takes ShapeRecord by value | 25f3b70 |
| s2-cg | shapes.h:28; flight.h:180; flashes.h:15; glow.h:57; scene.h:252; wander.h:63 | C.2 | reject | these are plain data the core hands across families and to the backend as the GPU holds it (scene.h: "the scene as the GPU will hold it"); Shapes and SceneDescription are read field by field by src/metal; making them classes is an architectural change across areas. Their invariants are now checked where they are used (I.6 rows: position(), schedule glow, .at() and P.6 throws) | — |
| s2-cg | flight.cpp:127-131 | C.12 | fix | Context has a constructor and deleted copy, so its references are never copied | 25f3b70 |
| s2-cg | obstacles.h:36; scene.cpp:474 | C.67 | fix | Obstacles: copy/move deleted, protected default constructor | 25f3b70 |
| s2-cg | motion.h:43; glow.h:47; flight.h:163; schedule.h:47 | Enum.7 | fix | underlying types dropped on the host-only enums | 25f3b70 |
| s2-cg | motion.h:44-45; glow.h:48-49; flight.h:164-167 | Enum.8 | fix | enumerator values dropped | 25f3b70 |
| s2-cg | graph_file.cpp:41-44; scene.cpp:46-49; schedule.cpp:118-123,131-136,177-180,218-224 | ES.1 | fix | ranges::none_of / copy_if / find / find_if / any_of | 25f3b70 |
| s2-cg | ES.3 (a) axis accessor x6 | ES.3 | fix | contracts::component(Float3, axis), host-only, in float3.h; wander, flight, swarm, shapes, scene use it | 25f3b70 |
| s2-cg | ES.3 (b) splitmix64 copy | ES.3 | fix | wander uses draw.h's splitmix64; its keying unchanged (bits identical, tests pass) | 25f3b70 |
| s2-cg | ES.3 (c) cross/length/normalized x2 | ES.3 | reject | Camera and Animation each do their own arithmetic at their own precision; float3.h's rule and file-mapping.md ("families depend on contracts, not on each other") keep vector math in its axis; a shared host math module is an architectural change outside this sweep | — |
| s2-cg | ES.3 (d) three box types | ES.3 | fix (in part) | animation::Extent is now contract 11's Box (both Animation's). shapes::Bounds stays: it is the Shape family's GPU-shaped bounding box (static_assert 24 bytes), and Shape may not depend on Animation's contract 11 | 25f3b70 |
| s2-cg | ES.3 (e) two TOML readers | ES.3 | reject | Frame graph and Scene content each read their own file (file-mapping.md); a shared TOML module would be a new family; both were fixed and tested for the read check | — |
| s2-cg | ES.3 (f) integer from node x3 | ES.3 | fix | Reader::integer(node, least, most, message) | 25f3b70 |
| s2-cg | ES.3 (g) triples parsed twice | ES.3 (P.9) | fix | read_unit_color reads and checks once | 25f3b70 |
| s2-cg | flight.cpp:34,44,45,53,56 | ES.7, NL.7 | fix | Vec3, to_vec, world_up, triple_at, put_triple | 25f3b70 |
| s2-cg | shapes.cpp:29/37,33/41,109-110 | ES.8 | fix | lo/hi/low/high gone: component() over .min/.max | 25f3b70 |
| s2-cg | flight.cpp (11 lines); swarm.cpp:30 | ES.10 | fix | one name per declaration | 25f3b70 |
| s2-cg | scene.cpp:227,252 | ES.12 | fix | read_unit_color, no inner `key` | 25f3b70 |
| s2-cg | scene.cpp (locals/params hiding namespaces), glow.cpp:22, transform.h:51,65, shapes.cpp:109-110, graph_file.cpp:42, scene.cpp:44,56 | ES.12 | fix | `description`, `all`, `texture_name`, `brightness`, `offset`/`to`/`factor`, `allowed_key`, `t` | 25f3b70 |
| s2-cg | shapes.cpp:99-100; swarm.cpp:30,41; wander.cpp:79,108; scene.cpp:761 | ES.20 | fix | std::array{} / initialized at declaration | 25f3b70 |
| s2-cg | animate.h:55,60; motion.h:49-50; glow.h:53-54; flight.h:150,155,181,214; wander.h:64; camera.h:83-86; scene.cpp:444; flight.cpp:200-202 | ES.20 | fix | default member initializers on every listed member | 25f3b70 |
| s2-cg | wander.cpp:73-75; scene.cpp:761-766 | ES.22 | fix | Wander{.anchor, .reach}; start from an immediately invoked lambda (ES.28) | 25f3b70 |
| s2-cg | graph_file.cpp:151; scene.cpp:207,286,328,901,929; frame_times.cpp:29 | ES.23 | fix | {} initialization; frame_times computes in Seconds | 25f3b70 |
| s2-cg | wander.cpp:79,108; flight.cpp:233,236; shapes.cpp (6); swarm.cpp:30,41 | ES.27 | fix | std::array | 25f3b70 |
| s2-cg | shapes.cpp:16 | SL.con.1 | fix | Placement::t is std::array<double, 3> | 25f3b70 |
| s2-cg | tone_map.h, noise.h, swirl.h, wood.h | ES.30 | fix | SERENITY_CONSTANT defined once in contracts/shared_layout.h, never undefined | 25f3b70 |
| s2-cg | bsdf.h, emitter.h, medium.h, surface_interaction.h, texture_reference.h, light.h + 4 | P.11 | fix | every shared header includes shared_layout.h and declares constants with SERENITY_CONSTANT; per-header #if blocks gone | 25f3b70 |
| s2-cg | flight.cpp:460,505; history.cpp:38-44 | ES.40 | fix | assignment out of the condition; rate by a switch; message built in named steps | 25f3b70 |
| s2-cg | scene.cpp:234,238,241-242,259-260,265-266; flight.cpp:515,521; scene.cpp:583,750; glow.h:66 | ES.45 | fix | messages format wood_least_ring/board, numeric_limits<uint32_t>::max(), swirl_most_vanes; flash_spacing named in flashes.h and used by flight, glow, reader (the "a second" words tied to it by static_assert) | 25f3b70 |
| s2-cg | flight.cpp:26; swirl.h:43; wood.h:81 | ES.45 claim | fix | circle's numbers named; wood_waver_octaves, wood_pores_seed, swirl_waver_octaves added in the core headers, the claim reworded to what the header holds; the shader literals are cross-area | 25f3b70 |
| s2-cg | flight.cpp (17 lines); glow.cpp:33; wander.cpp:82,84; thin_lens.cpp:72; png.cpp:16,26; scene.cpp:438,602 | ES.45 | fix | named: velocity_step, flight_most_steps, drift/swoop/transit ranges, waypoint_headroom, swoop_flash_at, jitter_span, least_weight/ratio, half_radians_per_degree, channels, numeric_limits max sentinels | 25f3b70 |
| s2-cg | flight.cpp:214-215; shapes.cpp:95-116; scene.cpp:195-197,834 | ES.49 | fix | static_cast; `(void)add_sphere` replaced by using its result | 25f3b70 |
| s2-cg | flight.cpp:433-449 | ES.86 | fix | while loop with explicit ++k / --k steps | 25f3b70 |
| s2-cg | thin_lens.cpp:61; graph_file.cpp (4); scene.cpp (9) | ES.87 | fix | `if (p)` / `if (!p)` | 25f3b70 |
| s2-cg | history.cpp:63 | ES.103 | fix | overflow checked before multiplying, std::overflow_error; test | 25f3b70 |
| s2-cg | history.cpp:56 | ES.104 | fix | index < since_ refused (HistoryError) before subtracting; test | 25f3b70 |
| s2-cg | scene.cpp:632,764; flight.cpp:586 | E.14 | fix | animation::Refusal (refusal.h); make_flight, make_wander, firefly_start throw it; reader and make_flights catch only it; test: an Obstacles' own invalid_argument propagates as itself | 25f3b70 |
| s2-cg | flight.cpp:97 | Con.5 | fix | constexpr velocity_step | 25f3b70 |
| s2-cg | scene.cpp (15 places) | T.42 | fix | `using NameIndex` | 25f3b70 |
| s2-cg | all 49 headers | SF.8 | convention | AGENTS.md "Guideline deviations" | 7755d04 |
| s2-cg | flight.cpp:423,462; scene.cpp (several); box.h:24,26 | SF.10 | fix | `<utility>`, `<functional>` included; box.h gets uint32_t from shared_layout.h, which it includes directly | 25f3b70 |
| s2-cg | every `#include "core/..."` | SF.12 | convention | AGENTS.md "Guideline deviations" | 7755d04 |
| s2-cg | flight.h:124-137; tone_map.h:68-79; surface_interaction.h:42-47 | NL.3 | fix | load timings to docs/research/2026-10-10-flight-load.md; half-float walk-through, M3 Max rounding and AgX figure to 2026-10-10-bloom-half-float.md; "from 48 before" dropped (already in 2026-10-09-medium-cost.md); headers keep the rule and a pointer; README table updated; scene.h's pointer updated | 25f3b70 |
| s2-cg | wood.h:64-71 | NL.3 | reject | not measurement: it derives the bound noise.h's step 1 requires (lattice coordinates within 2^31 from world_extent and the board/ring limits), a design fact of the same kind as frame_inputs.h's float spacing | — |
| s2-cg | graph_file.cpp:46-47; coated.h:74; flight.cpp:78,505; scene.cpp:220,862,469-470 | NL.4 | fix (in part) | each cited line rewrapped under 120 columns, the double blank removed, the continuation aligned. Adding and applying a .clang-format rejected: a repo-wide reformat touches every area other agents are editing in parallel (a large unstructured rewrite, CLAUDE.md 6.3) | 25f3b70 |
| s2-cg | scene.h:265,269,275,247; png.h:32 vs others | NL.8 | cross-area | renaming SceneDescription's fields and the Error types changes src/metal's readers and src/app / src/headless catch sites | — |
| s2-cg | flight.cpp:142 | NL.11 | fix | 1'000'000 (flight_most_steps) | 25f3b70 |
| s2-cg | flight.h:177; flight.cpp numbers[] | P.1 | fix | named index constants per behaviour (TransitAt, CircleAt, SwoopAt, DriftAt), read only by name; a std::variant rejected for now: it changes the public Segment the tests and the per-frame path read, for what names already give | 25f3b70 |
| s2-cg | scene.cpp:533,610 | P.1 | fix | contracts::scale(const Transform&) | 25f3b70 |
| s2-cg | frame_times.h:28-29 | F.8 citation | fix | "deterministic, no ambient input (I.1); not pure" | 25f3b70 |
| s2-cg | camera.h:52-53 | C.48 citation | fix | every Camera member has a default member initializer; a default Camera is invalid (tested) | 25f3b70 |

## s2-core-perf.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s2-perf | scene.cpp:484-490; shapes.cpp | CDSA.32 | fix | shapes::StillShapes places each still shape once (transform checked and widened once), per kind; same arithmetic, bit-identical answers; marbles load 3-4x faster (numbers above) | 25f3b70 |
| s2-perf | scene.cpp:474-505 | CACHE.4 (SIMD.2) | fix | per-kind arrays of what each test reads (sphere center+radius, box faces), scanned linearly; measured with the CDSA.32 row | 25f3b70 |
| s2-perf | scene.cpp:484-490,731; swarm.h; flight.h | CDSA.27 | reject | the flat exact scan is what CDSA.27 recommends at 51 still shapes; the index threshold is recorded in the research note (~1000 fireflies x 200 still shapes, ~1 s), not gated since no scene nears it; an early-exit squared-distance query rounds differently at the threshold and would re-draw every loop, for a sqrt per sphere | — |
| s2-perf | flight.cpp:141-144 | CDSA.21 | fix | named, derived, injectable most_steps (with s1 ES.46 row) | 25f3b70 |
| s2-perf | flight.cpp:527-536 | CDSA.9 | reject | Per.1/Per.6: measured animate() for 616 movers + glowers at 43 us median, under 0.3% of the frame; no measured reason for a second starts array | — |
| s2-perf | animate.h; motion.h; flight.h; glow.h | CACHE.3 | reject | same measurement as CDSA.9 (Per.1, Per.6) | — |
| s2-perf | flight.cpp:347-365,393-395,414,459 | MEM.1 | reject | profiled: allocation, strings and TOML 0.06% of the load (Per.6) | — |
| s2-perf | flight.cpp:473-517; scene.cpp (several) | MEM.1 (reserve) | reject | same profile, 0.06% (Per.6) | — |
| s2-perf | flight.cpp:542-592 | MEM.6 | reject | same profile; an injected pmr resource buys nothing measurable (Per.6) | — |
| s2-perf | scene.cpp:612-615 | CACHE.2 | reject | 616 small FlightParams copies: inside the 0.06% (Per.6) | — |
| s2-perf | scene.cpp:581,692 | CACHE.2 | reject | some 60 doubles per firefly, ~0.3 MB for 616: and the copy is the design (flashes.h: a glow depends on no motion kind); swarm_test pins it | — |
| s2-perf | scene.cpp:587,695 | COPY.3 | fix | push_back(std::move(glow)) at last use (ES.56); no speed claimed | 25f3b70 |
| s2-perf | flight.cpp:352,362 | COPY.7 | reject | initializer_list copies of at most two segments per transit: inside the 0.06% (Per.6) | — |
| s2-perf | flight.cpp:476-480 | COPY.7 | fix | lay() pushes the segment and sets start on the copy in place (part of the F.3 split); no speed claimed | 25f3b70 |
| s2-perf | scene.cpp:85-111 | COPY.9 | reject | error strings on the success path: inside the 0.06% (Per.6) | — |
| s2-perf | scene.cpp:203-206,282-284,323-326,892 | CDSA.10 | reject | name tables of tens of entries built once: inside the 0.06% (Per.6) | — |
| s2-perf | obstacles.h:36-45 | COPY.4 | fix | with C.67: copy/move deleted, protected default constructor | 25f3b70 |
| s2-perf | CMakeLists.txt; CMakePresets.json | GEN.4 | cross-area | ThinLTO is the build's (CMake owner); after CDSA.32 the hot load chain no longer crosses into shapes.cpp per shape | — |
| s2-perf | animate.cpp:9-18 | GEN.7 | reject | Per.1/Per.6: the whole animate() is 43 us median for 616+616; the two check passes are O(n) over 8-byte records; no measured cost | — |
| s2-perf | flight.h:120-122 (wander.h:47-53) | GDSA.2 | fix | flight.h states the level held (same toolchain, libm, flags) and why; a test pins one seed's loop (loop, segment and flash counts, two positions, exact bits; passes in release and debug) | 25f3b70 |
| s2-perf | frame_times.h:25-26 | TLM.6 | fix (core part) | header states a shown time names build type and resolution and is a window run, not a benchmark; the title is cross-area | 25f3b70 |
| s2-perf | sphere_light.h; light.h | GPU.4 (GPU.2) | reject | docs/research/2026-10-10-scene-block.md measured `device` 5% slower than `constant` for the scene block; the suggested `device const` contradicts that measurement, and a per-frame center/radius copy breaks sphere_light.h's design (no second copy of the center to fall out of step). GPU.10: remeasure with counters if lights become the bound | — |

## Rows from other reports that land in src/core

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s2-metal-host | contract structs | SL.con.4, COPY.6, LIFE.4 | fix (core part) | every shared layout (29 structs) asserts std::is_trivially_copyable_v on the host beside its size; the rule is in frame_constants.h's layout rules and shared_layout.h. bytes<T>, the casts and Bounds' offsetof checks are src/metal's | 25f3b70 |
| s2-shaders | emitter.h:18-22 (direct.metal.h:182) | I.1, I.5 | fix | contract 3 states u = (0, 0) draws the light's middle (a sphere's center) for every kind, and the sphere's cos t = 1 - u.x(1 - cos a) | 25f3b70 |
| s2-shaders | wood.h, swirl.h claims | ES.45 | fix (core part) | octave counts and the pores' seed offset named in the core; claims reworded; the shader literals are cross-area | 25f3b70 |
| s2-app | frame_times.h | TLM.6 | fix (core part) | see the s2-perf TLM.6 row | 25f3b70 |

## Counts

105 rows:

- fix: 82 (including 7 marked "in part" or "core part", whose remainder is
  a reject or a cross-area item stated in the row)
- convention: 2 (SF.8, SF.12)
- reject: 18
- cross-area: 3 rows (I.24 for the headless plan, NL.8, GEN.4); the
  cross-area list below also carries the remainders of four fixed rows

## Cross-area

| finding | file it needs | whose area |
|---|---|---|
| ES.45 (s2-shaders row 4 / s2-cg claim): use the new core constants instead of literals: `fbm(q, 3u, ...)` -> `tx::wood_waver_octaves`, `data.seed + 1u` -> `data.seed + tx::wood_pores_seed` (wood.metal.h:32,39); `fbm(..., 2u, ...)` -> `tx::swirl_waver_octaves` (swirl.metal.h:25) | src/metal/textures/wood.metal.h, swirl.metal.h | shaders |
| I.1/I.5: direct.metal.h:182's comment can cite contract 3's new u = (0, 0) rule (core/contracts/emitter.h) instead of "for a sphere" | src/metal/integrator/direct.metal.h | shaders |
| SL.con.4/COPY.6/LIFE.4 (host part): static_assert(std::is_trivially_copyable_v<T>) in bytes<T> and the reinterpret casts; offsetof checks for Bounds; scene_acceleration.cpp:11's "byte for byte" | src/metal/scene, src/metal/acceleration, src/metal/frame | metal host |
| I.24: plan_headless(first, samples, accumulates, scene_changes, time_frozen) and HeadlessPlan::sample(frame, s) want a parameter struct and named frame/sample types; the core signature and src/headless/main.cpp change together | src/core/frame/history.h + src/headless/main.cpp | headless (with core) |
| NL.8: SceneDescription's singular fields (rough, coated, absorbing) and the `Error` types in scene and output vs `XxxError` elsewhere; renaming touches every reader and catch site | src/core/scene/scene.h, src/core/output/png.h + src/metal/scene/*, src/app, src/headless | metal host, app, headless (with core) |
| GEN.4: ThinLTO (`-flto=thin`) on the Release preset, measured | CMakeLists.txt, CMakePresets.json | CMake owner |
| TLM.6: the window title shows GPU ms with only the resolution; name the build type (NDEBUG) and that it is a window run, not a benchmark (frame_times.h now states the obligation) | src/app/main.cpp | app |
