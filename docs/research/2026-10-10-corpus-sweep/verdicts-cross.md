# Verdicts: cross-area items (everything but tests/)

Branch `worktree-agent-afb467e957796c450`, from `sweep-merge` 0b0891b. Not
pushed or merged. Commits, oldest first:

| Commit | Subject |
|---|---|
| 626210a | Name PBR Neutral's numbers and use the textures' named ones, and cite the rules where they live |
| 7d1e229 | Pass the headless plan its run and its sample by name |
| 7855bf8 | Say what the toolchain pin holds, and that the headless renderer wants a new or empty directory |
| ac2767a | Name every exception type for its subject, and every kind's array for its kind |
| af0104b | Wrap the headless synopsis's new line within the header's width |
| 71386b7 | Refuse silent conversions, shadowed names and C-style casts in src/ |
| cea2b67 | Run every test under UndefinedBehaviorSanitizer, and record why AddressSanitizer cannot run |
| 5b8a704 | Read the path pass's accumulated image as the frame's radiance, and stop writing the mean twice |

## Table

| item | files | verdict | what was done | commit |
|---|---|---|---|---|
| ES.45 (core x-area 1, shaders x-area 2): shaders use the core's named fbm octaves and seed offset | src/metal/textures/wood.metal.h, swirl.metal.h | fix | `fbm(q, 3u, ...)` -> `wood_waver_octaves`, `seed + 1u` -> `seed + wood_pores_seed`, `fbm(..., 2u, ...)` -> `swirl_waver_octaves`; values unchanged | 626210a |
| ES.45 (shaders x-area 3): PBR Neutral's 0.04 / 0.08 / 6.25 | src/core/passes/tone_map.h, src/metal/passes/tone_map/tone_map.metal | fix | `neutral_f90` (F90 = 0.04), `neutral_toe` (2 F90 = 0.08), `neutral_toe_curve` (1 / (4 F90) = 6.25) named beside `neutral_start`; step 5 in the header written with the names and their derivation; the shader reads them | 626210a |
| I.1/I.5 (core x-area 2, shaders x-area 1): direct.metal.h cites contract 3's u = (0, 0) | src/metal/integrator/direct.metal.h | fix | `light_middle`'s comment cites core/contracts/emitter.h's rule (every kind maps u = (0, 0) to the light's middle) instead of "the sphere maps it; contract 3 does not say" (the core's half was 25f3b70) | 626210a |
| line over 120 columns (host x-area): scene_block.metal.h:52 | src/metal/scene/scene_block.metal.h | fix (already) | fixed by the shader sweep in 2137d2f; checked: no line in src/ over 120 now (two the renames made, scene.cpp:55 and tone_map.cpp:39, wrapped) | 2137d2f, ac2767a |
| wrong citation (host x-area): the 2^24 - 1 limit cited to metal/frame/accumulation.h | src/core/contracts/frame_constants.h, src/core/frame/history.h | fix | FrameConstants cites `frame::max_accumulated_frames` (core/frame/frame_inputs.h); history.h's pointer kept (it is where the backend asks History) and now says the limit is the core's | 626210a, 7d1e229 |
| I.24 (core x-area, host x-area): `plan_headless(first, samples, bool, bool, bool)` and `HeadlessPlan::sample(frame, s)` | src/core/frame/history.h/.cpp, src/headless/main.cpp | fix | `plan_headless(const HeadlessRun&)` with named fields, `sample(SampleOf{.frame, .sample})`; designated initializers at the one caller; the plan computed is the same. tests/history_test.cpp edited to compile (same assertions) | 7d1e229 |
| NL.8 (core x-area): SceneDescription's singular `rough`, `coated`, `absorbing` | src/core/scene/scene.h/.cpp, src/metal/scene/scene_block.h, scene_block.metal.h, scene_buffers.cpp, src/metal/materials/resolve.metal.h, src/metal/media/media.metal.h | fix | rule stated in scene.h: each kind's array is its kind's name, plural, adjectives included: `roughs`, `coateds`, `absorbings`; `gpu::SceneBlock` and the shaders' Materials/Media views the same. No layout change. tests/scene_test.cpp, tests/gpu/bsdf_test.cpp edited to compile | ac2767a |
| NL.8 (core x-area): exception types `Error` in four namespaces vs `XxxError` elsewhere | src/core/scene, src/core/output/png, src/app/window, src/metal/** (error.h and every throw), src/core/animation (refusal.h) | fix | one form, `<Subject>Error`: `scene::SceneError`, `output::PngError`, `app::WindowError`, `metal::MetalError`; `animation::Refusal`, a third form, is `MotionError` (refusal.h -> motion_error.h); comments naming them follow (graph_file.h's "an Error" -> GraphFileError). Tests that name the types edited to compile | ac2767a |
| GDSA.6 (shaders x-area 6): the path pass writes the mean twice | src/metal/passes/path/path.metal/.h/.cpp, src/metal/passes/bindings.h, src/metal/frame/frame_images.h/.cpp, src/metal/frame/renderer.h/.cpp | fix (measured; kept for the memory, not the time) | the renderer binds the accumulated image as the frame's radiance; FrameImages takes `Radiance{image, accumulated}` and makes no radiance image for an accumulating light pass (123 MB at 3456 x 2234); the kernel loses its radiance texture and write. Re-measured: 0.06 ms of 28.2 at the median of ten alternated pairs, within the noise; the 0.30 ms did not reproduce. Note docs/research/2026-10-10-path-radiance-write.md; numbers kept out of the headers. tests/gpu/tone_map_test.cpp edited to compile | 5b8a704 |
| MEMORY.md's toolchain lock text (host x-area, lead) | docs/process/MEMORY.md, README.md | fix | names what cmake/toolchain.json pins: Xcode, the SDK the presets compile against, the Metal compiler, Apple clang as C, C++ and Objective-C compilers, and CMake. README said "CMake 3.25 or later", which the pin refuses: now 4.2.3, and Python 3.9 or later as CMakeLists.txt finds it | 7855bf8 |
| headless refuses a non-empty `--out` (host x-area, lead) | src/headless/options.cpp, options.h, README.md | fix | the usage text and options.h's synopsis say DIRECTORY must be new or empty; README says it beside `make headless` | 7855bf8, af0104b |
| P.12: `-Wconversion -Wsign-conversion -Wshadow -Wold-style-cast` | CMakeLists.txt | fix | `SERENITY_SOURCE_WARNINGS` = `SERENITY_WARNINGS` + the four, on every target in src/ (serenity_core, serenity_metal, serenity_app_options, serenity, serenity_headless_options, serenity-headless). src/ already built clean under them in both presets: no source change was needed. Test targets keep `SERENITY_WARNINGS` (tests agent) | 71386b7 |
| P.12: sanitizer preset (ASan + UBSan) | CMakePresets.json, Makefile, cmake/ubsan_ignorelist.txt, docs/research/2026-10-10-sanitizers.md | fix (UBSan); ASan blocked | `native-sanitize` (configure, build, test presets; `make test-sanitize`): debug, `-fsanitize=undefined -fno-sanitize-recover=all`, dependencies included. Core tests: no report. GPU tests: one report, metal-cpp's `NS::Referencing::release()` called through a null pointer by `SharedPtr` assignment (Objective-C's message to nil); exempted by those two members' mangled names only, and a null call of ours is still reported (checked with a probe program). `ctest --preset native-sanitize` 5/5. ASan not in the preset: its runtime deadlocks before `main` on macOS 26.6.2 (in a five-line program; Apple clang 17 and Homebrew clang 21 alike; sandbox ruled out), trace in the note | cea2b67 |
| SL.con.4/COPY.6/LIFE.4 host part (core x-area) | src/metal/scene, acceleration, frame | fix (already) | checked present: `is_trivially_copyable_v` in `bytes<T>`, FrameArray, the block, the renderer's ring copies; Bounds' `offsetof` checks against MTL::AxisAlignedBoundingBox (657d709 and the host sweep) | 657d709 |
| GEN.4 ThinLTO (core x-area, CMake owner) | CMakeLists.txt, CMakePresets.json | reject | as the host sweep decided (verdicts-host.md): the frame is GPU-bound (28 ms GPU, the host records in microseconds); LTO changes only host code generation, a build-configuration change (CLAUDE.md 6.5) with no measured gain (Per.1, Per.6) | - |
| TLM.6 window title (core x-area, app) | src/app/main.cpp | fix (already) | the title says it is a live window, not a benchmark (host sweep) | 80787a5 |
| BsdfSample carrying eta (shaders x-area 4, optional) | src/core/contracts/bsdf.h | reject | bsdf_eta() is correct as it is (the shader sweep says so); BsdfSample is 32 bytes with no spare field, so carrying eta grows every sample's struct (a contract layout, static_assert) to save a lookup that is not measured to cost anything (Per.1, Per.6) | - |
| CameraData unit right/up, ToneMap exposure scale (shaders x-area 5, optional) | src/core/contracts/camera.h, core/passes/tone_map.h | reject | the shader sweep recommends it only with a measurement, and none shows a need: the path kernel is ray-tracing bound with the ALU 13% busy (docs/research/2026-10-10-path-kernel-counters.md), so one normalize per camera ray and one exp2 per pixel are not where its time goes (Per.6, GPU.10) | - |
| AGENTS.md records SF.8 and uint loop indices (shaders x-area 8) | AGENTS.md | fix (already) | both in "Guideline deviations" (7755d04) | 7755d04 |
| tests/gpu path-numbers probe placement (shaders x-area 7); allocation-counting test, argument_table_test header (host x-area) | tests/ | tests agent | listed below | - |

## For the tests agent

Tests my changes need, or that the cross-area rows name (I added none):

1. GDSA.6 (5b8a704): `FrameImages(device, submission, Radiance::accumulated,
   Bloom::none/pyramid)` makes no radiance image (`radiance()` null after
   `prepare()`) and, with the pyramid, every level; `PathPass::record` no
   longer needs `resources.radiance` and still throws `MetalError` without an
   accumulated image or counter; a path graph's renderer holds no radiance
   image of its own (only the accumulated image and the pyramid are
   resident). The path tracer's images are already checked unchanged by the
   existing GPU tests.
2. I.24 (7d1e229): history_test.cpp was rewritten mechanically to
   `plan_headless({.first, .samples, .accumulates, .scene_changes,
   .time_frozen})` and `sample({.frame, .sample})`; same assertions. A test
   of `HeadlessRun`'s defaults is optional.
3. NL.8 renames (ac2767a): test case names still say the old type in
   tests/gpu/frame_errors_test.cpp:68 ("refused by Error"); the
   animation_test case name was updated to MotionError. Type names in test
   code were changed only where they no longer compiled.
4. Warnings (71386b7): add `-Wconversion -Wsign-conversion -Wshadow
   -Wold-style-cast` to serenity_tests and serenity_gpu_tests (use
   `SERENITY_SOURCE_WARNINGS`, or fold the four into `SERENITY_WARNINGS`
   once tests/ is clean), and fix what they raise there.
5. Sanitizer (cea2b67): `ctest --preset native-sanitize` passes today; keep
   it passing, and consider naming it in the verification steps.
6. tests/gpu/tone_map_test.cpp:114 writes PBR Neutral's 0.08 / 6.25 / 0.04
   as literals in its reference; keep them as an independent oracle or read
   `passes::neutral_toe`, `neutral_toe_curve`, `neutral_f90` (626210a).
7. From the other sweeps' cross-area lists: the path-numbers probe in
   bsdf_probe.metal / bsdf_test.cpp moves if a sampler test file is added
   (shaders x-area 7); an allocation-counting test around
   `Renderer::record()` and `SceneAcceleration::update()` for the MEM.9
   claims, and tests/gpu/argument_table_test.cpp's header (Metal does state
   that it snapshots the table at each dispatch) (host x-area).

Test files I edited only to keep them compiling: tests/history_test.cpp,
tests/scene_test.cpp, tests/swarm_test.cpp, tests/animation_test.cpp,
tests/flight_test.cpp, tests/png_test.cpp, tests/gpu/bsdf_test.cpp,
tests/gpu/frame_errors_test.cpp, tests/gpu/frame_test.cpp,
tests/gpu/history_test.cpp, tests/gpu/library_test.cpp,
tests/gpu/lifetime_test.cpp, tests/gpu/path_test.cpp,
tests/gpu/preview_test.cpp, tests/gpu/tone_map_test.cpp.

## Results

- `cmake --build --preset native-release` and `native-debug`: clean under
  -Werror, src/ with the four new flags.
- `./build/native-release/tests/serenity_tests`: 113 cases, 2617
  assertions, pass.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1
  ./build/native-release/tests/gpu/serenity_gpu_tests`: 88 cases, pass.
- `ctest --preset native-release`, `native-debug`, `native-sanitize`: 5/5
  each.

## Measurements

GDSA.6, the path tracer on scenes/marbles.toml at 3456 x 2234, 200 frames
in flight, GPU start to end from commit feedback, median of frames 21-199;
two binaries alternated, each run started after no serenity process ran and
`ioreg` reported the GPU at most 5% busy for six seconds (another process
outside this work kept it near 100% for stretches). Ten pairs over three
sessions:

| Session | Before (ms) | After (ms) |
|---|---|---|
| 1 | 28.535, 28.490, 28.503 | 28.426, 28.446, 28.439 |
| 2 | 28.227, 28.202, 28.222 | 28.439, 28.146, 28.155 |
| 3 | 28.254, 28.224, 28.219, 28.214 | 28.130, 28.156, 28.239, 28.238 |

Median difference -0.06 ms (0.2%), seven of ten pairs faster, within the
noise; the shader sweep's 28.75 -> 28.46 did not reproduce. Kept for the
123 MB image and the per-pixel write it removes; a single commit (5b8a704)
if the lead prefers to drop it.
