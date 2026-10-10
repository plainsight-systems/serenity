# Verdicts: tests (tests/**, tests/gpu/kernels, the tests' CMakeLists)

Branch `worktree-agent-ab5e12c6e36fe17e3`, from `sweep-merge` at 0b0891b.
Not pushed or merged. No file under src/ is changed.

| Commit | Subject |
|---|---|
| 073a19f | Make the tests that passed when the code was wrong able to fail |
| 956d1a2 | Test what the suite left untested, through one probe runner and shared helpers, under stricter warnings |

Each finding was checked against the code at 0b0891b before acting (the
earlier fix agents had already added lifetime_test, frame_errors_test,
programs_test and touched options_test, library_test, tone_map_test,
acceleration_test).

## Sabotage record

Every HIGH fix, and every new coverage test, was shown to fail with the
code under test broken in the worktree (never committed; each file restored
with `git checkout` and the tree rebuilt; `git status src` clean after
each). "Old" is the test as it was at 0b0891b.

| Finding | Sabotage (src, uncommitted) | Old test | New test |
|---|---|---|---|
| H1 resize | frame_images.cpp: pyramid levels neither released nor remade on a resize | passes (alpha 255 always) | fails: the 97x61 frame differs from a never-resized renderer's |
| H2 non-finite | path.metal: sample = NaN (bit pattern) when within.x < 0.01 (1% of samples) | passes: every mean within tolerance | fails in all 11 convergence tests on `non_finite_samples() == 0` (e.g. 41764) and only there |
| H3 conductor | conductor.metal.h schlick(): (a) float3(f0.x) for all channels; (b) `f0` with no angle term | passes (f0 = 1) | fails `worst < 1e-4` at all three wo, both sabotages |
| H4 dielectric | (a) fresnel.metal.h: Schlick's approximation in place of the equations; (b) dielectric.metal.h: refracted = incident direction; (c) transmitted_eta swapped | (a), (b) pass (normal incidence); (c) fails | (a) fails: 41843 wrong reflections, 1006733 wrong refractions, share off by 0.0016 against 5 sigma 0.00097; (b) fails ~1e6 wrong refractions; (c) fails |
| H5 Approx | glow.cpp: rhythm flash half as long (lit 4.0%) | passes: Approx(0.08).epsilon(0.05) accepts 2.6%..13.4% (verified by running) | fails: 0.039999 vs 0.08 at 1% |
| gap 3 block | scene_buffers.cpp: woods and swirls given each other's arrays | resolve test fails only through its swirl check | probe fails on the woods and swirls fields (9 of 12 words each); frame fails: green on the swirl (1341), the wood not brown (49 pixels) |
| gap 4 sampler | sampler.metal.h: key from pixel.x / 2 (neighbours share a stream) | passes | fails: neighbour correlation 0.50 vs bound 0.0195, every dimension |
| gap 5 light | sphere_light.metal.h: cos t = 1 - u.x^2 (1 - cos a) | cone test passes | fails: chi^2/dof 2844 vs bound 1.44 |
| MEM.9 allocations | renderer.cpp record(): one `::operator new(8)` | (no test) | fails in all three cases (1 allocation) |
| gap 6 headless | headless/options.cpp written(): doubling writes every frame | (no test) | fails: frame-000009.png written |

Also measured, not a change: before attribution, Metal (and, with
MTL_DEBUG_LAYER, its validation layer) allocates through operator new on
the recording thread (6 to 15326 calls a frame); attributed to their first
caller outside libc++/libSystem, record() makes 0 allocations of ours in
every case (still and moving scenes, path and preview graphs), so the MEM.9
claims in renderer.h and scene_acceleration.h hold.

## s1-tests.md

| report | file:line | rule ID | verdict | what was done / sabotage result | commit |
|---|---|---|---|---|---|
| s1 H1 | gpu/frame_images_test.cpp:72 | 3.6 facade | fix | Resize checked byte for byte against a renderer that never saw the larger size; images compared by count of differing bytes. Sabotage above | 073a19f |
| s1 H2 | gpu/path_test.cpp:60-72 | 3.3 | fix | Rig::render() checks `non_finite_samples() == 0` after every converged render; render_once() and the frame-k test too. Sabotage above | 073a19f |
| s1 H3 | gpu/bsdf_test.cpp:270-304 | 3.6 | fix | Brass f0: evaluate(brass)/evaluate(f0 = 1) against Schlick's in double over every lobe cell, at 0, 53 and 80 degrees, all channels (worst < 1e-4); mean weight per channel against the per-channel albedo. Sabotage above | 073a19f |
| s1 H4 | gpu/bsdf_test.cpp:306-338 | 3.6 | fix | 30 and 60 degrees from outside, 30 from inside: reflected share against exact Fresnel (5 sigma), every reflection's mirror direction, pdf F and weight 1, every refraction's Snell direction, pdf 1 - F and weight 1/eta_t^2 (1/2.25 in, 2.25 out). Sabotage above | 073a19f |
| s1 H5 | flight_test.cpp:229 and every Approx | Approx scale | fix | `.scale(0).epsilon(0.01)`. Every one of the 81 Approx without scale(0) audited: nonzero values now relative with the epsilon stated; zeros stay absolute, stated; a unit vector's components and positions near 0 keep scale 1 with a comment (camera_test failed relative on a 4e-4 component, as expected). Sabotage above | 073a19f |
| s1 M1 | animation_test.cpp:92 | Approx | fix | `.scale(0).epsilon(0.02)` | 073a19f |
| s1 M2 | gpu/CMakeLists.txt:42 | P.12 | fix | ENVIRONMENT MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER_ERROR_MODE=assert on serenity_gpu_tests and the new headless test | 956d1a2 |
| s1 M3 | swarm_test.cpp:4,196-273; core/animation/flight.cpp:759 | I.1, CP.9 | needs src | make_flights takes its count from hardware_concurrency(), no parameter: a test cannot vary it. The header's "whatever the threads" claim removed from the test; the file states what is not tested. TSan preset is CMakePresets.json (lead). See needs-src | 956d1a2 |
| s1 M4 | gpu/tone_map_test.cpp:90,119-125,145 | 3.6 | fix | The reference writes out 6 levels, 65504, 0.5, 0.125, 0.76, 0.15 and F90 0.04 (offset 2 F90, 1/(4 F90)) from the header's words; nothing read from core/passes/tone_map.h but the pass's own array size | 956d1a2 |
| s1 M5 | gpu/path_test.cpp:5-9,106-117 | NL.3 | fix | Header no longer claims the furnace shows depth; integrating sphere at rho 0.9, L 20 (0.45, within the display), 2048 frames: a cut at 8 bounces keeps 57%, at 16 82% | 956d1a2 |
| s1 M6 | bsdf_test.cpp:237,381; path_test.cpp:391 | 3.6 | fix | Lambert weights and albedo, coat weights, coated furnaces, the glow's (0.6, 0.3, 0.1) and the filled glass's blue all per channel | 073a19f, 956d1a2 |
| s1 M7 | scene_test.cpp:203-248 | 3.6 | fix | Every mistake checks `s.toml:<n>:`, here and in the motion, wood, lens, coated/swirl/media, swarm and flight tests; line numbers stated in comments | 956d1a2 |
| s1 M8 | png_test.cpp:20-48 | 3.6, P.8, 4.2 | fix | 5x3 image of a distinct value per channel, decoded with stb_image (the same pinned stb, SYSTEM include, test target only) and compared whole; temp file named by pid, removed by RAII | 956d1a2 |
| s1 L1 | coated_test.cpp:39 | Approx | fix | `.scale(0).epsilon(0.005)`; the Fresnel integrand from support/references.h | 073a19f |
| s1 L2 | flight_test.cpp:119-120 | 3.6 | fix | z checked too | 073a19f |
| s1 L3 | bsdf_test.cpp:213-218 | 3.6 | fix | Bins under 20 pooled into one Pearson bin; if the pool expects under 20, it may hold at most 3x + 10 | 073a19f |
| s1 L4 | path_test.cpp:324-329 | 3.6 | needs src | The renderer gives the accumulated float image to no reader but its passes; the test says so in its comment and compares the displayed bytes by count. Needs a float readback (needs-src) | 956d1a2 |
| s1 L5 | swarm_test.cpp:257; kernels/trace_probe.metal:2 | NL.3 | fix | Test name no longer claims "after every thread ends"; the probe names acceleration_test | 956d1a2 |
| s1 gap 1 | conductor colored Fresnel | coverage | fix | = H3 | 073a19f |
| s1 gap 2 | dielectric oblique / inside | coverage | fix | = H4 | 073a19f |
| s1 gap 3 | scene block mapping | coverage | fix | New gpu/scene_block_test.cpp + kernels/scene_block_probe.metal: every word of every element of all 19 arrays through the block, shader-typed stride; a preview frame reading wood, swirl, emissive and medium; resolve now reads through the block. Sabotage above | 956d1a2 |
| s1 gap 4 | principle 2 | coverage | fix | path_test: frames 0-4 started over then 5-7 accumulated, against 5-7 alone on a fresh renderer, and 5 alone differs from 6; new gpu/sampler_test.cpp: frame 5's numbers alone equal frame 5's after frames 0-4, uniform means, no correlation across dimension, frame or neighbour (5/sqrt(N)). Sabotage above | 956d1a2 |
| s1 gap 5 | sphere-light sampling | coverage | fix | light_test: 2^20 emitter draws binned 16x16 in (cos t, phi) over the cone, chi^2; every draw inside the cone, sample pdf and pdf() both 1/solid angle. Sabotage above | 956d1a2 |
| s1 gap 6 | headless main | coverage | fix | New tests/test_headless.sh (ctest `headless`): doubling/last files and names, printed paths, 16x8 headers, same bytes twice, status 1 + message + nothing written for an unknown option, a broken graph, a directory holding a frame, and a scene whose light overflows (samples not finite). Sabotage above. A GPU twin in path_test checks Film counts a computed infinity | 956d1a2 |
| s1 gap 7 | Offscreen/FrameArray/StaticArrays/frame_times | coverage | fix | frame_errors_test: read_rgba short and long refused, exact accepted; {16, 0} refused; FrameArray of no bytes refused (both copies); StaticArrays with no arrays has no array 0. frame_times add() precondition was already tested (frame_times_test.cpp:37) | 956d1a2 |
| s1 gap 8 | make_flights thread count | coverage | needs src | = M3 | - |

## s2-tests.md

| report | file:line | rule ID | verdict | what was done | commit |
|---|---|---|---|---|---|
| s2 | camera/texture/shape/film/light/argument_table/library/acceleration dispatch copies | ES.3, F.10, P.11 | fix | gpu/support/probe_runner.h (ProbeRunner, Binding, Buffers::read) used by bsdf, sampler, camera, texture, shape, film, light, scene_block tests. Kept their own dispatch, stated in each header: argument_table_test (rebinding between dispatches is its subject), library_test (the Library alone, no backend), acceleration_test (binds a structure and records the update in the trace's encoder) | 956d1a2 |
| s2 | probe record layouts | SF.3, I.4 | fix | gpu/kernels/probes.h: every record shared, integer counts and indices (shape, seed, found, grid), sizes asserted both sides, trivially-copyable on the host; bsdf_probe.h folded in; `Probe` now BsdfDraw | 956d1a2 |
| s2 | camera_probe.metal:18; camera_test 1600/900 | ES.45, I.4 | fix | CameraImage bound beside the camera; one `image` constant in the test | 956d1a2 |
| s2 | light/acceleration/bsdf buffers | P.7, ES.65 | fix | REQUIRE after every newBuffer/newTexture; runner REQUIREs encoder and table | 956d1a2 |
| s2 | preview_test.cpp:272-294 | SF.22 | fix | core_lit_inside_glass, tinted_bead moved to support/scene_text.h (inline, one definition), with their stretches named | 956d1a2 |
| s2 | camera_text/sky/linear/sRGB/Vec/camera ray/Fresnel copies | ES.3, F.10 | fix | support/scene_text.h, support/references.h (srgb8, linear_of, camera_direction, pixel_of, fresnel_reflectance, fresnel_from_air), support/vector.h (Vec3) | 956d1a2 |
| s2 | error_for/headless_error/error_of/why/says; contains | ES.3, F.10, NL.8 | fix | support/text.h: error_of<E>(callable), contains(string_view), replaced(); each file keeps a one-line wrapper naming its parser | 956d1a2 |
| s2 | flight/swarm fixtures, distance helper | ES.3, NL.8 | fix | support/ball_on_floor.h: the scene, the shape indices, distance_to_still, BallAndFloor. gpu/history_test's floor-and-firefly scene is a different scene, kept | 956d1a2 |
| s2 | within-file duplication | ES.3, F.10 | fix | path: mean_channel/mean_red, furnace(), Patch/middle_of, through_filled_glass; motion: animated()/still(); light: one runner; texture: colors<Data>(), differing(); bsdf: cell_direction, directional_albedo, weight, weighs, mean_weight, Sampled reuse; flight: FlightText | 956d1a2 |
| s2 | replace(find(x), N, y) | ES.45, ES.3 | fix | tests::replaced(), which REQUIREs the find | 956d1a2 |
| s2 | options_test prefix and limits | ES.45, NL.11 | fix | headless_args() prefix; numeric_limits<uint64_t>::max(), frame::max_accumulated_frames, max/2 | 956d1a2 |
| s2 | animation/flight/swarm/scene fixture literals | ES.45 | fix | wandering params, reach/speed/seed; body/clearance/volume; written_shapes, firefly_radius, swarm_clearance, count; flights, fireflies. flight's 0.55 named swoop_flash_at in the test: flight.h does not name it (needs src) | 956d1a2 |
| s2 | image sizes, centers, counts | ES.45 | fix | size.width / 2, middle_of(), probed pixels, frames, resized, side/c, dim_from, boards/per_board, count | 956d1a2 |
| s2 | byte sizes; binding indices | ES.45 | fix / reject | sizeof everywhere. Named binding enums rejected: the bindings are positional, stated once (probe_runner.h) and listed in each kernel's comment; an enum on the host would not check the kernel's literal [[buffer(n)]] either (I.4 asks for a check, which neither form gives) | 956d1a2 |
| s2 | rgba_bytes(Extent) x10 | ES.3 | fix | Offscreen::rgba_size() via tests::read_back() | 956d1a2 |
| s2 | FrameInputs builder, schedule strings | ES.3, NL.8 | fix | tests::frame_at(), preview_graph(), path_graph() (gpu/support/rendering.h); CPU history_test keeps its own frame_at of the same name | 956d1a2 |
| s2 | long test bodies | F.3, F.2 | fix | tone_map reference split into exposed/pyramid_down/pyramid_up/composite; bsdf check into check_consistent/check_histogram; light and motion through helpers | 956d1a2 |
| s2 | tone_map_test.cpp:141,214 | F.20, F.21 | fix | `Toned {image, base}` returned; Pyramid enum | 956d1a2 |
| s2 | film/bsdf/acceleration adjacent params | I.24 | fix | Folding{pixels, samples, starts}; bsdf Parameters{.alpha, .ior}; TraceRay from two Vec3 | 956d1a2 |
| s2 | acceleration 7 params; bsdf_resolve 12 buffers | I.23 | fix | Traced struct; resolve reads through the scene block (4 bindings) | 956d1a2 |
| s2 | bool arguments of src APIs | I.4 | needs src | plan_headless(…, bool, bool, bool), LiveHistory(bool), History::join(…, bool), Accumulation::prepare(…, bool), ShapeTransforms(…, bool): the core and host owners' signatures | - |
| s2 | Gpu::run, bytes<T>, down13 | T.10 | fix | GpuBytes concept; std::invocable<double, double> | 956d1a2 |
| s2 | texture_test.cpp:79-83 | ES.20, ES.48 | fix | seed is a uint32 field of TexturePoint, no float bits | 956d1a2 |
| s2 | bsdf_probe.metal:31-37 | ES.20, ES.22 | fix | BsdfDraw aggregate-initialized | 956d1a2 |
| s2 | NS::Error never read | I.10, P.7 | fix | tests::reason(error) in INFO before every argument-table REQUIRE | 956d1a2 |
| s2 | wait_until_complete result dropped | NL.8 | fix | `(void)` everywhere it is dropped | 956d1a2 |
| s2 | tone_map_test.cpp:222-247 | ES.26 | fix | shared_texture() makes its own descriptor | 956d1a2 |
| s2 | tone_map_test.cpp:256/260 | ES.8 | fix | table_descriptor | 956d1a2 |
| s2 | `frame`, `scene`, `cos`, `floor` locals | ES.12, NL.19 | fix | description, slot, round, shown, cos_wi, cos_h, floor_material, traced; -Wshadow on | 956d1a2 |
| s2 | function-style casts | ES.49, ES.46 | fix | static_cast; -Wold-style-cast and -Wconversion on | 956d1a2 |
| s2 | mixed signedness, `1 << 20` | ES.100, ES.41 | fix | typed counts; -Wsign-conversion on | 956d1a2 |
| s2 | path_test.cpp:95-96 | ES.106, ES.102 | fix | Patch REQUIREd inside the image before the loop | 956d1a2 |
| s2 | animation_test.cpp:84-85 | P.2, ES.31 | fix | std::numbers::pi | 956d1a2 |
| s2 | thin_lens_test.cpp:106,111 | ES.31 | fix | numeric_limits<float>::infinity() | 956d1a2 |
| s2 | tone_map_test.cpp:55,281 | P.2 | fix | `using Half = _Float16`, commented as the extension until std::float16_t | 956d1a2 |
| s2 | SERENITY_*_DIR macros | ES.31 | fix | configure_file'd test_paths.h (scenes_dir, graphs_dir as filesystem paths), via the serenity_test_support INTERFACE target | 956d1a2 |
| s2 | <algorithm>, <cstdint>, <optional>, <vector>, <span>, <utility> | SF.10 | fix | Every test checked by script for names used without their header, and headers included unused | 956d1a2 |
| s2 | history_test.cpp:5, tone_map_test.cpp:15 | SF.10 | fix | removed | 956d1a2 |
| s2 | C arrays | ES.27, SL.con.1 | fix / convention | std::array in host code; `float u[8]` and padding in probes.h shared with Metal (convention, AGENTS.md) | 956d1a2 |
| s2 | motion_test.cpp:31-35 | SL.io.3 | fix | std::to_chars, its result checked | 956d1a2 |
| s2 | graph_file_test:76-81; library_test:56-61 | ES.1 | fix | CHECK_THROWS_WITH_AS(…, doctest::Contains(…), E). animation_test keeps try/catch: it must catch Refusal before invalid_argument, which CHECK_THROWS cannot | 956d1a2 |
| s2 | `.find() != npos` | ES.1, ES.3 | fix | tests::contains | 956d1a2 |
| s2 | raw loops | ES.1 | fix | std::ranges::count_if/all_of/any_of/adjacent_find/min_element, std::equal, starts_with, transform_reduce | 956d1a2 |
| s2 | animation_test.cpp:33-35 | ES.40 | fix | contracts::component() | 956d1a2 |
| s2 | const that could be constexpr | Con.5 | fix | constexpr params, string_view | 956d1a2 |
| s2 | non-const locals | Con.1, Con.4 | fix | const added | 956d1a2 |
| s2 | several names per declaration | ES.10, NL.21 | fix | one each | 956d1a2 |
| s2 | positional aggregates with padding | ES.23, ES.22, P.1 | fix | designated initializers (BoxData, ShapeRecord, WoodData, SwirlData, FlightJob, Folding, WanderParams) | 956d1a2 |
| s2 | swarm_test.cpp:171-179 | C.12 | fix | BallAndFloor holds a pointer | 956d1a2 |
| s2 | path_test.cpp:48-73 | C.2, C.8 | fix | Rig a class with accessors; GPU history_test's Rig likewise | 956d1a2 |
| s2 | class vs struct fakes, Probe x3, index types, using vs qualified | NL.8, ES.8 | fix | fakes `class … final : public`; distinct record names; schedule_test and graph_file_test one form each | 956d1a2 |
| s2 | stale comments | NL.2, NL.3 | fix | acceleration probe, preview ray, path floor comment, trace_probe's test | 956d1a2 |
| s2 | long lines, layout | NL.4, NL.17 | fix / reject | every line within 120 columns, double blanks gone. `.clang-format` rejected: clang-format is not in the pinned toolchain (cmake/toolchain.json), so its output would depend on whichever is installed (4.3, P.12); a pinned formatter is the lead's to add | 956d1a2 |
| s2 | kernels/bsdf_probe.h:1 | SF.8 | convention | `#pragma once` (AGENTS.md) | - |
| s2 | gpu/CMakeLists.txt metallib `smoke` | NL.19 | fix | named `probes` | 956d1a2 |
| s2 | device/library/pipeline per probe call | GDSA.17, GPU.6, P.9 | fix | one ProbeRunner per test case | 956d1a2 |
| s2 | bsdf duplicate 1M draws and densities | GPU.1, P.9 | fix | check_sampling_follows_pdf returns its Sampled draws and cells, reused | 956d1a2 |
| s2 | push_back without reserve | P.9 | fix | reserve where the count is known | 956d1a2 |
| s2 | exp2 / eta in loops | P.9 | fix | hoisted (coated's now in fresnel_from_air) | 956d1a2 |
| s2 | png_test temp file | R.1, P.8 | fix | = M8 | 956d1a2 |

## Cross-area items for the tests

| report | item | verdict | what was done | commit |
|---|---|---|---|---|
| verdicts-host MEM.9 | allocation-counting test around Renderer::record() / SceneAcceleration::update() | fix | New gpu/allocation_test.cpp: replaced operator new, counted on the recording thread, attributed by backtrace/dladdr to the first caller outside libc++ and libSystem; this program's image counts. record() of a moving, blinking scene (update() inside) and of a still one, path and preview graphs: 0 allocations of ours each frame, under validation and without. The counter shown to count (::operator new from the test) and to catch one added to record() | 956d1a2 |
| verdicts-host | argument_table_test.cpp header said Apple leaves rebinding unsaid | fix | Header quotes MTL4ComputeCommandEncoder.h's snapshot rule and says the test shows it on this machine | 956d1a2 |
| verdicts-shaders 7 | path-numbers probe in bsdf_probe.metal | fix | kernels/sampler_probe.metal and gpu/sampler_test.cpp | 956d1a2 |
| verdicts-host (lead) | -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast | fix (tests) | On both test targets through serenity_test_support; tests/ builds clean under -Werror in native-release and native-debug | 956d1a2 |

## Noticed, not changed

- bsdf_test's chi-square over its 512 x 512 equal-area grid reports
  chi^2/dof ~3 when a sharp lobe sits at the grid's pole (wo = (0.17, 0,
  0.985) about n = x); the same wo turned into the x-y plane passes. A
  limit of the test's binning at the poles, which the existing comment
  already steers lobes away from; the new colored-f0 test keeps its lobes
  on the equator.

## Needs src

1. `animation::make_flights` (core/animation/flight.h:242, flight.cpp:759):
   take the worker count as a parameter (default the machine's) so a test
   can run 1, 2, 7 and 64 workers and compare (s1 M3, gap 8; I.1). And a
   ThreadSanitizer preset in CMakePresets.json (lead).
2. A read-back of the accumulated float image (Renderer or Accumulation, a
   test-facing accessor or an RGBA32F offscreen target), so determinism
   and principle 2 can be compared in floats, not displayed bytes (s1 L4).
3. core/animation/flight.h: name `swoop_flash_at` (flight.cpp:79) in the
   header, so flight_test reads it rather than repeating 0.55 (ES.45).
4. Bool parameters to enums or option structs (I.4): plan_headless(first,
   samples, bool, bool, bool), LiveHistory(bool), History::join(…, bool),
   Accumulation::prepare(…, bool), ShapeTransforms(…, bool).
5. A pinned clang-format in cmake/toolchain.json before any .clang-format
   is adopted (NL.17; lead).

## Counts

s1-tests.md: 26 rows (5 high, 8 medium, 5 low, 8 gaps): fix 23, needs src
3 (M3, L4, gap 8).
s2-tests.md: 60 rows: fix 55, fix in part and reject in part 2 (binding
names; .clang-format), needs src 1 (I.4 bools), convention 1 (SF.8), fix
with a convention part 1 (C arrays in shared layouts).
Cross-area: 4, all fixed.

## Tests

- `cmake --preset native-release` / `native-debug`, both built with the
  test targets under -Wconversion -Wsign-conversion -Wshadow
  -Wold-style-cast -Werror: clean.
- `./build/native-release/tests/serenity_tests`: 113 cases, 2527
  assertions, pass.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1
  ./build/native-release/tests/gpu/serenity_gpu_tests`: 101 cases, pass
  (1.87M assertions; per-sample CHECKs became counts, from 26M).
- `ctest --preset native-release` and `--preset native-debug`: 6/6 pass
  (serenity_tests, boundaries, boundaries_fire, toolchain_fires,
  headless, serenity_gpu_tests; the GPU and headless tests under Metal
  validation).
