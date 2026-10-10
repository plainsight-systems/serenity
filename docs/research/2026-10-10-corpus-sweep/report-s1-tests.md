## Test suite review: serenity `tests/` (branch `marbles`, read-only)

I found 5 high-severity problems: in each, a test passes when the code is wrong. There are also 8 medium and 5 low problems. No test in the suite times anything, so the C++ performance corpus did not apply. C++ Core Guidelines IDs are cited where a rule applies.

### High (a test passes when the code is wrong)

**H1. `tests/gpu/frame_images_test.cpp:72`** — §3.6 (facade).
- The check `any_of(rgba, b > 0)` cannot fail. The tone-map pass always writes alpha 1.0 (`tone_map.metal:131`), so every pixel's alpha byte is 255.
- "The next frame is whole" is never checked: a black or garbage image after the resize passes.
- Fix: render the same input at 97x61 on a fresh renderer and compare byte for byte, as lines 42–65 already do for the four frames.

**H2. `tests/gpu/path_test.cpp:60-72` (`Rig::render`), and every convergence test in that file** — §3.3.
- Samples that are not finite are dropped from the pixel mean (film_test proves this).
- No furnace, NEE, medium or coat test checks `renderer.non_finite_samples() == 0`. A bug that turns 1% of paths into NaN biases the mean by the survivors and still passes the 2–3% tolerances.
- Fix: add `CHECK(renderer.non_finite_samples() == 0)` in `Rig::render` after the wait.

**H3. Conductor Fresnel is never exercised** — `bsdf_test.cpp:270-304` and `preview_test.cpp:345-356` use only f0 = [1, 1, 1].
- At f0 = 1, `schlick()` (`conductor.metal.h:44`) returns 1 at every angle.
- A conductor that ignores f0 or its color passes every test. The resolve test checks `Bsdf.color` but not `evaluate()`.
- Fix: test a colored f0 (brass) and check the per-channel mean weight against the integral of `evaluate()·cos`, and `evaluate()` against Schlick computed in double at several angles.

**H4. Dielectric tested only at normal incidence and past the critical angle** — `bsdf_test.cpp:306-338`, `preview_test.cpp:248-262`. The test name claims "Fresnel chooses reflection".
- At normal incidence, exact Fresnel, Schlick and a constant F0 all agree, and the direction of eta does not matter.
- So these pass when wrong: Fresnel at oblique angles, Snell's refracted direction, the eta² weight on exit, and refraction from inside below the critical angle. The furnace tests conserve energy whatever F is, so they do not catch these either.
- Fix: probe wo at 30° and 60° from outside and at 30° from inside. Check the reflected fraction against `fresnel_from_air`, the refracted direction against Snell's law, and the weight against 1/eta² and eta².

**H5. `tests/flight_test.cpp:229`** — `Approx(0.08).epsilon(0.05)` has no `.scale(0)`, so the tolerance is 0.05·(1 + 0.08) = ±0.054.
- It accepts any lit fraction from 2.6% to 13.4%. A flash of 0.2 s or 0.65 s instead of 0.4 s passes.
- The result is deterministic, so fix with `.scale(0).epsilon(0.01)`.

### Medium

**M1. `tests/animation_test.cpp:92`** — `Approx(0.3).epsilon(0.02)` gives ±0.026, which is ±8.7% relative, not the 2% the test reads as. Fix: `.scale(0).epsilon(0.02)`.

**M2. GPU tests never run under Metal validation** — `tests/gpu/CMakeLists.txt:42`, `CMakePresets.json` testPresets (P.12).
- There is no `MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION` or error-mode setting anywhere.
- Probe kernels are bound to 16-byte stand-in buffers for empty arrays (`bsdf_test.cpp:58`, `texture_test.cpp:45`). An out-of-bounds read there returns garbage silently.
- Fix: `set_tests_properties(serenity_gpu_tests PROPERTIES ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER_ERROR_MODE=assert")`, or a second ctest entry with those settings.

**M3. Parallel flights: thread count is ambient and the tests hunt races by timing** — `swarm_test.cpp:4,196-273`; `src/core/animation/flight.cpp:569` (I.1, CP.9; profile §4.2, §4.4).
- `make_flights` takes its thread count from `std::thread::hardware_concurrency()`, so "whatever the threads" (header, line 4) is never varied.
- `for (run < 5)` at line 228 can only catch a race by luck. There is no ThreadSanitizer preset.
- Fix: make the worker count a parameter, test it with 1, 2, 7 and 64 workers, and add a TSan preset.

**M4. `tests/gpu/tone_map_test.cpp:90,119-125,145`** — the CPU reference uses the implementation's own constants (`down_middle`, `down_corner`, `neutral_start`, `neutral_desaturation`, `bloom_ceiling`).
- A wrong constant in `core/passes/tone_map.h` matches itself.
- Fix: pin the literal values in the test (0.5, 0.125, 0.76, 0.15, 65504).

**M5. `tests/gpu/path_test.cpp:5-9,106-117`** — the Lambert furnace uses a convex sphere viewed from outside.
- Every path is camera → sphere → sky: one bounce. Roulette only starts at the 4th surface (`path.metal.h:65`), so it never runs here.
- The header's claim that "a depth limit … shows it here" is false for this test. Only the integrating sphere (line 119) tests depth, and at 3% tolerance a cap at 16 or more bounces would pass it.
- Fix: correct the claim, and tighten the integrating-sphere test or raise ρ to 0.9.

**M6. Only one color channel is checked** — `bsdf_test.cpp:237-238` (albedo {0.5, 0.7, 0.9}, only `.x`), `bsdf_test.cpp:381` (coated), `path_test.cpp:391` (the comment says (0.6, 0.3, 0.1); only red is read). A channel swap, or using `color.x` for every channel, passes. Fix: check all three channels.

**M7. `tests/scene_test.cpp:203-248`** — the name says each mistake is refused "naming the file and the line", but only lines 204 and 234 check a line number. Fix: assert `s.toml:<n>:` for each case.

**M8. `tests/png_test.cpp:20-48`**
- The pixels are a uniform 0x80 and only the signature and IHDR are read, so stride, channel order and row order in the `stbi_write_png` call are untested.
- Fixed file names in `temp_directory_path()` (TMPDIR, an undeclared input) collide if both presets run at once.
- Fix: write a distinct per-pixel gradient, decode it (stb_image) and compare; use a unique per-run path.

### Low

- **L1.** `coated_test.cpp:39` — `Approx(0.092).epsilon(0.01)` is ±12% on the test's own helper. Use `.scale(0).epsilon(0.005)`.
- **L2.** `flight_test.cpp:119-120` — loop periodicity is checked on x and y, not z.
- **L3.** `bsdf_test.cpp:213-218` — chi² drops bins expecting fewer than 20 samples instead of pooling them, so samples landing where the pdf is near zero are not counted. Pool them into one bin.
- **L4.** `path_test.cpp:324-329` — determinism is compared on 8-bit display bytes, which hides float differences below 1/255. Compare the accumulated float image.
- **L5.** `swarm_test.cpp:257` claims "after every thread ends", which the test cannot observe. `trace_probe.metal:2` names motion_test; the probe belongs to acceleration_test.

### Coverage gaps (behaviour in `src/` with no test)

1. Colored conductor Fresnel and per-channel `evaluate()` (H3).
2. Dielectric at oblique incidence and from inside (H4).
3. Scene block mapping (`metal/scene/scene_block.h`, `scene_buffers`): no render goes through the block with wood, swirl, emissive or medium arrays, so swapped addresses would pass. The resolve test binds those arrays directly. The `static_assert` checks only the struct's size.
4. Principle 2, "any frame can be rendered again exactly, alone": nothing renders frame k on its own and compares it with frame k inside a sequence, and nothing checks that the sampler's dimensions and frames are independent.
5. Sphere-light sampling against its pdf: `light_test` checks only the cone's edge and one pdf value. The NEE tests use lights too small to expose a sampler that is not uniform. A chi² like the BSDF one is needed.
6. Headless `main` (`src/headless/main.cpp:74`): the non-finite failure path, which frames are written to disk, and the exit status.
7. Error throws in `Offscreen` (zero size, `read_rgba` with the wrong size), `FrameArray`, `StaticArrays`, and the `add()` precondition in `frame_times.h`.
8. Thread-count independence of `make_flights` (M3).

### Checked and fine

- Statistical GPU tests draw their numbers from hashed sample indices or the frame-indexed sampler, so they are deterministic, not flaky. Their thresholds sit about 4–6σ out.
- `.scale(0)` is used correctly in `bsdf_test`, `path_test` and `preview_test`. The tight `epsilon(1e-5)` and `epsilon(1e-6)` checks (acceleration t, thin-lens dot products, shape positions) really are tight.
- Every `(void)wait_until_complete(...)` is fine: it throws on GPU failure.
- GPU tests fail rather than skip when there is no device.
- `error_of` helpers return "" when parsing succeeds, so a parse that wrongly succeeds fails its `contains` check.
- Tone map: a CPU reference in double with half-float rounding, plus a check that the comparison can see the bloom at all (line 350).
- The motion tests compare animated and hand-placed renders byte for byte.
- Film counts non-finite samples correctly at a size that is not a multiple of the SIMD width.
- The submission-misuse, history (CPU and GPU), options and graph-file tests are thorough.
- No test reads a clock: `frame_times` is driven by numbers.
- The coated white furnace (0.3% at 2^20 samples) and the BSDF consistency checks are sound.