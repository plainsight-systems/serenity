# Verdicts: host C++ (src/metal non-shader, src/app, src/headless), build, tools, scripts

Branch `worktree-agent-adbfe687dcfaf2639`, from `marbles` at c0bc7d1. Commits:

| Commit | Group |
|---|---|
| 6e085e5 | A. Lifetime: nothing freed while the GPU uses it |
| 5defed7 | B. Frame-path errors refused before `begin()`, refusals change nothing |
| 657d709 | C1. Compile-time checks: scene block list, trivially copyable byte copies |
| e91ace0 | C2. One binding-index header for kernels and passes |
| 6f8a4d8 | C3. One argument table; miscounts and miscitations corrected |
| 80787a5 | F. App and headless programs |
| 23ba080 | G. Build, tools, scripts (ctest checks, pins, SDL SYSTEM, movie step) |
| 065bbb1 | D. Measurements out of headers into docs/research; barrier measured |
| dd8cacc | E. Backend refactors (support.h, typed interfaces, Con.2, C.49, ...) |
| 34b244a | The submission's feedback poll measured, its header corrected |

Measurements: see the end of this file.

## s1-metal-host.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s1-metal-host #1 | device/submission.cpp:62-71; renderer.h:175; every owner | C.31, R.1, GPU.9 | fix | `Submission::wait_idle()` (noexcept, bounded); `keep_resident()` returns a `Resident` that on destruction waits, removes its allocation from the set, then releases it; Renderer and Presenter wait in their destructors (pipelines, argument table, layer set are outside the set); owners: Offscreen, FrameArray, StaticArrays, SceneBuffers, SceneAcceleration, NonFinite, ToneMapPass, Accumulation, FrameImages, the renderer's ring. Uncertainty resolved by measurement: the residency set retains (retain count 1 -> 2 on add). Test `tests/gpu/lifetime_test.cpp`: an exception with two path frames in flight; without the wait the GPU faults (MTL4CommandQueueErrorDomain error 1), with it the frames complete and `finish()` is clean, under the validation layer | 6e085e5 |
| s1-metal-host #2 | renderer.cpp:185-187, :237 | E.4, E.14 | fix | camera checked in `Renderer::prepare()`; glower targets checked at construction, so `animate()` cannot throw mid-frame; header states that only protocol/Metal failures can throw after `begin()`. Tests: `frame_errors_test.cpp` (no camera refused with nothing begun, run continues; bad glower refused at construction) | 5defed7 |
| s1-metal-host #3 | renderer.h:141-148 | no-facades | fix | cites MTL4ComputeCommandEncoder.h's snapshot sentence; the per-slot ring of tables collapsed to one table (frame_test's frames-in-flight test covers it) | 6f8a4d8 |
| s1-metal-host #4 | accumulation.cpp:15, 39-43 | E.4 | fix | history joined on a copy, kept only once the image exists; history reset when the old image is gone; size checked before the old image is released. Also found: Metal aborts the program (failed assertion, even without the debug layer) on a texture side > 16384 rather than returning nil, so `check_texture_size()` (device.h) refuses such sizes by Error in Offscreen, FrameImages, Accumulation, Presenter. Tests: refused oversized remake keeps image and history; Offscreen refuses 16385 and 0 | 5defed7 |
| s1-metal-host #5 | renderer.cpp:270-272 | I.5, I.6 | fix | `render_to_window` throws Error if the drawable's texture is not the presenter's size, before `begin()`. Test with a window-less CAMetalLayer whose drawable size is changed behind the presenter | 5defed7 |
| s1-metal-host #6 | submission.h:167-170 | I.5, I.6 | fix | `release_resident()` removed; release goes through `Resident`, which waits for the GPU itself, so the precondition holds by construction; a release while a submission is open marks it uncommittable (tested) | 6e085e5 |
| s1-metal-host #7 | scene_acceleration.cpp:80-81, 90, 110 | GPU.9 | fix | still scene's scratch reset after the waited build; header says so. Not observable through the interface (no allocation query that is not timing-dependent); verified by reading and by the GPU tests passing | 5defed7 |
| s1-metal-host #8 | scene_acceleration.h:57-59 | CDSA.29 (miscited) | fix | ordering cites GPU.8; CDSA.29 kept for "built once" | 6f8a4d8 |

## s1-app.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s1-app #1 | app/main.cpp:76-80; headless/main.cpp:43-46 | R.1, E.6, C.13, GPU.7 | fix | fixed in the owners (row s1-metal-host #1), so both mains are safe with their declaration order unchanged; their headers now say so | 6e085e5, 80787a5 |
| s1-app #2 | CMakeLists.txt:105; Makefile | MEMORY.md lock; Test the Contract | fix | ctest runs `boundaries` (check_boundaries.sh), `boundaries_fire` (test_check_boundaries.sh) and `toolchain_fires`; the two that touch src/ hold a RESOURCE_LOCK | 23ba080 |
| s1-app #3 | tools/check_boundaries.sh:20-21 | MEMORY.md boundary | fix | `(MTL[A-Z0-9]*\|NS\|CA)::`, `Metal[A-Za-z]*/`, IOSurface, CoreVideo, `@import`; probes added for MTLFX, MTL4FX, MPS, IOSurface, CoreVideo, @import | 23ba080 |
| s1-app #4 | CMakeLists.txt:12-26; check_toolchain.py:31; CMakePresets.json | CLAUDE.md 4.1, 4.3 | fix | project enables C, CXX, OBJC; all three compilers checked AppleClang and pinned (`cc`, `objc`); presets set `CMAKE_OSX_SYSROOT=macosx` and the check verifies the SDK CMake resolved (compile lines now carry `-isysroot .../MacOSX26.2.sdk`); `cmake` pinned (4.2.3); Python found by `find_package(Python3 3.9)`, not PATH, with a floor rather than an exact pin (it runs only the check and review scripts and builds nothing); probes for every new pin, a fake compiler/cmake, a wrong SDK root and an empty one | 23ba080 |
| s1-app #5 | Makefile:66 | determinism | fix | `printf "%.17g"`: 0.016666666666666666 parses to exactly 1.0/60.0 (checked), the headless default | 23ba080 |
| s1-app #6 | Makefile:43-46; headless/main.cpp:48 | principle 1 | fix | `headless::prepare_output()` creates the directory or takes an empty one, refuses one holding anything (OptionsError); `make headless` removes its own old `frame-*.png` first. Tests in `tests/programs_test.cpp`; checked by hand on the binary | 80787a5 |
| s1-app #7 | app/main.cpp; headless/main.cpp; CMakeLists.txt:170-172 | SF.10 | fix | includes added; `serenity_app_options` links `serenity_core` instead of a raw include directory | 80787a5 |
| s1-app #8 | scripts/post_commit_review.py:29,33 | CLAUDE.md 4.1 | fix | `git -c core.quotepath=false show --no-ext-diff --src-prefix=a/ --dst-prefix=b/` | 23ba080 |

## s2-metal-host.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s2-metal-host | renderer.cpp:277-284 | P.8, R.1 | fix | `render_to_offscreen` drains a pool of its own; header states it. Not observable through the interface; verified by reading | 5defed7 |
| s2-metal-host | scene_buffers.cpp:24-27, 34-60; scene_block.h:92 | P.5 | fix | one constexpr `std::to_array<Entry>` list of {field, bytes function}; static_asserts: count from the list against the block's size, every entry non-null and its own field (both shown to fire by breaking the list) | 657d709 |
| s2-metal-host | renderer.h:77-78 | NL.2 | fix | points at metal/passes/bindings.h | e91ace0 |
| s2-metal-host | path.h:42-44 | NL.2 | fix | 48 bytes | 6f8a4d8 |
| s2-metal-host | test_pattern.h:35-36 | NL.2 | fix | 32 bytes | 6f8a4d8 |
| s2-metal-host | submission.h:86-87 | NL.2 | fix | "at start-up and when a resize remakes an image" | 6e085e5 |
| s2-metal-host | submission.h:92-94; submission.cpp:140-156 | Per.14, COPY.7, MEM.9 | fix (comment) / reject (code) | comment states the real allocations (options object, heap-copied block and its captures). Reuse rejected: an options object cannot shed a handler, so it cannot be reused per slot with a new one; a few allocations against a 27.7 ms frame is not a measured cost (Per.1, Per.6) | 6f8a4d8 |
| s2-metal-host | renderer.h:158; scene_acceleration.h:83-84 | MEM.9 | fix (cross-area) | the check is an allocation-counting test around `record()`; a test of an existing claim belongs to the tests agent (listed below) | - |
| s2-metal-host | submission.cpp:85-92 | CP.40, Per.30, CONC.4 | reject (code) / fix (comment) | measured: every settle sleeps (200 of 200, 297-314 sleeps per 200 frames), yet GPU time and wall time a frame are the same (27.2 ms / 27.5-27.6 ms): the poll falls while the GPU runs the frame in flight, so a blocking wait on a second event has no measured gain (Per.6). The header now counts the poll and names the alternative; numbers in docs/research/2026-10-09-pass-costs.md. The interval is named (`feedback_poll`) | 34b244a, dd8cacc |
| s2-metal-host | submission.cpp:142-153 | E.12 | fix | the feedback lambda is `noexcept` | 6f8a4d8 |
| s2-metal-host | offscreen.cpp:21 ... presenter.cpp:23 | C.31 | fix | every owner releases what it made resident (Resident), Presenter takes its layer's set off the queue | 6e085e5 |
| s2-metal-host | renderer.cpp:74-75; tone_map.cpp:41; non_finite.cpp:11-12; scene_buffers.cpp:81; FrameArrays | GPU.9 | reject | GPU.9 is about steady-state and transient allocation; these are a dozen allocations made once at start-up (the renderer's ring is now a FrameArray, one fewer kind). Load time measured 408-448 ms for the whole start-up including the scene read and pipeline builds; no measured cost to suballocation (Per.6) | - |
| s2-metal-host | submission.cpp:218-219, 226-227 | GPU.6 | reject | GPU.6 batches tiny GPU work per frame; residency-set commits happen at start-up, at a resize (after a drain) and at teardown, never per frame; no measured cost (Per.6) | - |
| s2-metal-host | frame_array.cpp:40,45; non_finite.cpp:36-38; scene_acceleration.cpp:128,137; renderer.cpp:202 | I.5, I.6, SL.con.3 | fix | one check everywhere: FrameArray (both copy counts), NonFinite::address, SceneAcceleration::update/resource, Renderer::record throw Error for a slot >= frames_in_flight. Tested | 5defed7 |
| s2-metal-host | static_arrays.h:47; frame_images.h:88 | I.5, E.14 | fix | check and throw Error; tested for StaticArrays | 5defed7 |
| s2-metal-host | renderer.cpp:230; scene_acceleration.cpp:101 | ES.65, I.12 | fix | null encoder throws Error. Not testable (Metal does not return null on demand); after begin() it leaves the submission open, which the header documents | 5defed7 |
| s2-metal-host | submission.h:173 | C.9 | fix | `queue()` replaced by `add_residency_set()` / `remove_residency_set()` | 6e085e5 |
| s2-metal-host | presenter.cpp:17, 26-29 | C.41 | fix | constructor refuses a size Metal makes no image of (zero included); tested | 5defed7 |
| s2-metal-host | renderer.cpp:199,208; tone_map.cpp:45; scene_buffers.cpp; frame_array.cpp; static_arrays.cpp; shape_transforms.cpp; light_glows.cpp; scene_acceleration.cpp | SL.con.4, COPY.6, LIFE.4 | fix (host sites) | `static_assert(is_trivially_copyable_v)` at each memcpy/byte-view site in my files; byte views go through one `FrameArray::view<T>()`; Bounds checked against MTLAxisAlignedBoundingBox by size and both corners' offsets. The contract headers' own asserts are the core agent's (cross-area) | 657d709 |
| s2-metal-host | submission.h:70 | CP.3 (miscited) | fix | R.20, CP.32 | 6f8a4d8 |
| s2-metal-host | frame_images.h:50-51 | Per.6 | fix | measured (docs/research/2026-10-09-pass-costs.md): without the barrier a frame takes 44.7-48.3 ms against 27.8; header says the overlap is no gain | 065bbb1 |
| s2-metal-host | frame_images.h:75; renderer.cpp:122; accumulation.h:62; shape_transforms.h; light_glows.h; frame_array.h | I.4, I.24 | fix (part) / reject (part) | `FrameImages(…, Bloom)` (the radiance flag dropped: always true outside one test, which now makes the radiance image too); `FrameArray::Copies {one, per_frame}` replaces the 1-or-2 count and its error path. Rejected for the single-bool `moves`/`scene_changes`/`glowing` parameters: I.24 is about adjacent parameters of one type, and these are not adjacent to another bool | dd8cacc |
| s2-metal-host | passes' .cpp; tone_map.cpp | ES.45, P.5 | fix | `metal/passes/bindings.h`, read by the kernels' parameter lists and the passes; renderer's table sizes static_asserted against it. AIR byte-identical before and after (compiled at the same path) | e91ace0 |
| s2-metal-host | path.cpp:10-12; preview.cpp:10-12 | P.3 | fix | messages name every checked resource | e91ace0 |
| s2-metal-host | describe x4, pools, align, texture descriptor, dispatch x5, ring | ES.3, F.10 | fix | `metal/device/support.h`: `describe`, `scoped_pool`, `buffer_alignment`/`align_up`, `make_private_texture`, `dispatch_per_pixel`; ring is a FrameArray | dd8cacc |
| s2-metal-host | device.h:50 ... scene_acceleration.h:114-115 | Con.2 | fix (writers) / reject (handles) | writers non-const: `FrameArray::bytes/view`, `ShapeTransforms::transforms`, `LightGlows::glows`, `SceneAcceleration::update`. Rejected for `Device::handle`, `Library::handle`, `Offscreen::texture`, `Accumulation::texture`, `FrameImages::radiance/bloom`: they observe the owner's state; the handle is how metal-cpp takes every object (non-const pointers throughout its API), and a const pointer would need a const_cast at each Metal call (ES.50) | dd8cacc |
| s2-metal-host | device.cpp:16-19; submission.cpp:39,91; presenter.cpp:15; offscreen.cpp:25; renderer.h:220 | ES.45 | fix | family table of pairs over a range-for; `residency_capacity`, `feedback_poll`, `drawable_count`, `bytes_per_pixel`; `std::optional<std::size_t>` for the first cross-frame pass | dd8cacc |
| s2-metal-host | device.cpp:16-18; submission.cpp:155; scene_acceleration.cpp:75 | SL.con.1, ES.45 | fix | std::array, passing `.data()`/`.size()` | dd8cacc |
| s2-metal-host | every header | SF.8 | convention | - | - |
| s2-metal-host | renderer.h:3-7 ... frame_resources.h:11 | SF.10 | fix | includes added; frame_resources.h no longer includes scene_buffers.h; scene_block.h's host branch includes <stdint.h> for the global `uint32_t` it names | dd8cacc |
| s2-metal-host | accumulation.cpp:27; frame_images.cpp:61; scene_acceleration.cpp:110; reinterpret_casts | ES.48 | fix | `(void)` casts removed (the functions are not `[[nodiscard]]`, so nothing needs discarding); one `reinterpret_cast`, in `FrameArray::view<T>()` | 657d709, dd8cacc |
| s2-metal-host | frame_images.cpp:55,71; submission; renderer.cpp:197; shape_transforms.cpp:9; scene_buffers.cpp:33; scene_acceleration.cpp:56 | ES.8, NL.19, ES.7 | fix | `pool`, `settled_in_flight`, `Slot::settled_through`, ring slot offset gone with the FrameArray, `nonempty()`, `Block`, `metal_device` | 657d709, dd8cacc |
| s2-metal-host | renderer.h:189 etc.; non_finite.cpp:41; light_glows.h:39 | ES.12 | fix | `begun`, `total`, `glowing` | dd8cacc |
| s2-metal-host | renderer.h:210-214 | NL.16 | fix | `Step` and `Prepared` declared before the data members | dd8cacc |
| s2-metal-host | tone_map.h:41; renderer.h; preview.h:79-80; accumulation.h:66-67; scene_block.h:31; frame_resources.h:46-53 | NL.3, NL.4 | fix | reflowed; double blank line removed; the garbled sentence rewritten; the duplicated inline comment folded into the block comment | 065bbb1, dd8cacc |
| s2-metal-host | accumulation.cpp:13; renderer.cpp:201-205 | ES.22, ES.23 | fix | immediately invoked lambda for the join; designated initializers for FrameResources' always-present fields | 5defed7, dd8cacc |
| s2-metal-host | device.cpp:35-40; presenter.cpp:11; submission.cpp:33-58 | C.49 | fix | init-list helpers (`system_default`, `describe_device`, `retained`, `make_queue`, `make_residency_set`, `make_event`, `load`, `make_argument_table`) | dd8cacc |
| s2-metal-host | tone_map.cpp:79; submission.cpp:198 | ES.40, ES.86 | fix | `below` counts down with a plain condition; drain's `first` named | dd8cacc |
| s2-metal-host | device.cpp:10; submission.cpp:85,97; offscreen.cpp:31; MTL::Size(...) | ES.64, ES.23 | fix | braces | dd8cacc |
| s2-metal-host | observers | F.6 | fix | noexcept on `handle`, `info`, `texture`, `size`, `radiance`, `next_sequence`, `has_completed`, `NonFinite::count/counter`, `non_finite_samples` | dd8cacc |
| s2-metal-host | frame_images.cpp:32; static_arrays.cpp:15 | F.4 | fix | `half()` constexpr noexcept; `align_up` constexpr in support.h | dd8cacc |
| s2-metal-host | passes' record(encoder*); renderer target; frame_images make; submission | F.23, I.12 | reject | `not_null` is GSL's, which is not a dependency (adding one is a dependency change for an annotation, CLAUDE.md 6.5); metal-cpp's every call takes pointers, so references would be dereferenced back at each use; null is checked once where Metal hands the pointer over (encoder, drawable texture, record's arguments) | - |
| s2-metal-host | path.h:47; preview.h:85; display.h:35; test_pattern.h:41 | I.23 | fix | the unused `const Device&` removed; constructors `explicit` | dd8cacc |
| s2-metal-host | device/error.h:23 | C.52 | fix | inheriting constructor | dd8cacc |
| s2-metal-host | library.cpp:35-39 | R.1, E.19 | fix | `std::unique_ptr` with a `dispatch_release` deleter | dd8cacc |
| s2-metal-host | library.h:39 | SL.str.5 | fix | `span<const std::byte>`; the generated header exposes the bytes as `span<const std::byte>` over a constexpr `std::array<unsigned char, N>` | 23ba080 |
| s2-metal-host | submission.cpp:15; submission.h:185-186 | I.4 | fix | `std::chrono::milliseconds timeout`, `microseconds feedback_poll`; `Feedback::gpu_start/gpu_end` are `frame::Seconds` | dd8cacc |
| s2-metal-host | device.cpp:16-18; renderer.h:218-219 | C.1 | fix | pairs; `Step {kind, pass}` | dd8cacc |
| s2-metal-host | renderer.cpp:93-97; static_arrays.cpp:43 | ES.1, ES.71, ES.77 | fix | `std::ranges::find_if`/`any_of`; the `continue` inverted | dd8cacc |
| s2-metal-host | scene_acceleration.cpp:57-64,93 | P.9 | fix | the build-only residency set is made only for a still scene | 5defed7 |
| s2-metal-host | submission.cpp; renderer.cpp; passes | GEN.7 | reject | GEN.7 marks cold paths where profiling shows code layout matters; these throws are off a per-frame path that costs microseconds against a 27.7 ms GPU frame; no measurement shows a cost (Per.1, Per.6) | - |
| s2-metal-host | static_arrays.cpp:41; renderer.cpp:243; tone_map.cpp:70 | ES.107 | reject | the indices subscript `std::span`/`std::array`/`std::vector`, whose sizes and subscripts are `std::size_t`; a signed index would mix signed and unsigned at each use or need casts (ES.100, ES.102), and there is no GSL `index`. Loops that need no index are range-for | - |

## s2-app.md

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s2-app | CMakeLists.txt:41-42 vs 73-78 | P.12, 4.1 | fix | `SYSTEM` on SDL's FetchContent_Declare; SDL now included with `-isystem` | 23ba080 |
| s2-app | window.h:28-29, 44-46; main.cpp:75 | I.4 | fix | `app::Points` for the window's size in points; Extent stays pixels; header explains both | 80787a5 |
| s2-app | headless/options.h:68 | NL.2 | fix | cites `frame::max_accumulated_frames`, core/frame/frame_inputs.h | 80787a5 |
| s2-app | headless/options.h:45 | NL.2 | fix | cites core/frame/history.h | 80787a5 |
| s2-app | window.h:15 | NL.2 | fix | true again: the clock no longer uses SDL | 80787a5 |
| s2-app | clock.cpp:3,7,10-11 | SL.2, ES.1 | fix | `std::chrono::steady_clock`; clock moved into `serenity_app_options` and tested (monotonic, from 0) | 80787a5 |
| s2-app | clock.cpp:11 | ES.45, ES.48, P.1 | fix | chrono subtraction, no cast or literal | 80787a5 |
| s2-app | clock.h:19-20 | NL.2 | fix | GPU instants are host time from another origin | 80787a5 |
| s2-app | app/options.h:28-29 | NL.2 | fix | claim narrowed | 80787a5 |
| s2-app | headless/options.cpp:17-18 | NL.2, NL.3 | fix | comment matches the parameters | 80787a5 |
| s2-app | window.h:70, window.cpp:21 | NL.2 | fix | comment says SDL_Quit shuts every subsystem, and the program starts no other | 80787a5 |
| s2-app | window.h:34-35 | Per.6 | fix | claim dropped | 80787a5 |
| s2-app | app/main.cpp:68; headless/main.cpp:36 | ES.42, ES.100 | fix | `arguments(argc, argv)` returns an empty span for argc < 1 | 80787a5 |
| s2-app | app/options.cpp:37-40; headless/options.cpp:20-24, 37-40 | E.28 | fix | `std::from_chars` for integers and doubles (libc++ here has both); tests that a sign, space, trailing text or hex float is refused | 80787a5 |
| s2-app | app/options.cpp; headless/options.cpp | ES.86 | fix | an `Arguments` cursor; no loop index stepped in the body | 80787a5 |
| s2-app | headless/options.h:102 | I.24 | fix | `written(Write, RunFrame{.after_first, .frames})` | 80787a5 |
| s2-app | headless/options.cpp:46 | I.24 | fix | `side(text)` returns `optional<uint32_t>`; the caller names the option and the whole value | 80787a5 |
| s2-app | headless/main.cpp:54-55 (core/frame/history.h:124) | I.24 | fix (cross-area) | `plan_headless`'s signature is the core's | - |
| s2-app | headless/options.cpp:149-150 | P.6, 3.1 | fix | `throw std::logic_error`; tested with `static_cast<Write>(3)` | 80787a5 |
| s2-app | app/main.cpp:52-56 | ES.27, SL.con.1, SL.io.3, SL.4 | fix | `std::format` | 80787a5 |
| s2-app | headless/main.cpp:83-84 | ES.27, SL.con.1, SL.io.3, ES.45 | fix | `std::format("frame-{:06}.png", index)` | 80787a5 |
| s2-app | app/main.cpp:116; headless/main.cpp:87,92 | SL.io.3 | fix | `std::cerr` / `std::cout` | 80787a5 |
| s2-app | mains; headless/options.h:91,93 | ES.45 | fix (mains) / reject (options defaults) | `initial_window`, `title_period`, ms through `chrono::duration<double, milli>`, `bytes_per_pixel` in Offscreen. The option defaults (1/60 s, 1920x1080) are member initializers whose meaning the header states beside them, the named-constant form ES.45 asks for | 80787a5 |
| s2-app | headless/main.cpp:49 vs offscreen.cpp:25-26 | ES.3 | fix | `Offscreen::rgba_size()` | 80787a5 |
| s2-app | app/options.cpp vs headless/options.cpp; three error classes | ES.3 | reject | each program's command line is its own library by design (CMakeLists: each parses its command line in a library of its own), so a shared parser would couple the window to the headless renderer; with from_chars each parse is a few lines. The error types now share one name per program (next row) | - |
| s2-app | app/options.h:37; headless/options.h:79; window.h:39 | C.52 | fix | inheriting constructors | 80787a5 |
| s2-app | app vs headless options | NL.8 | fix | `headless::Error` renamed `headless::OptionsError`, as the window's | 80787a5 |
| s2-app | window.cpp:43-49 | C.49 | fix | `window_(open(...))`, `view_(make_view(...))` in the init list | 80787a5 |
| s2-app | window.h:48-51 | COPY.5 | fix | moves stay deleted; the header states the real reason (one video subsystem, one Window for the program's life, a moved-from Window would own nothing) | 80787a5 |
| s2-app | window.cpp:15-24, 26-37 | C.21 | fix | `SdlVideo` deletes its moves explicitly; `MetalView` replaced by `unique_ptr<void, DestroyView>` | 80787a5 |
| s2-app | window.h:77-79 | R.5 | fix | `SdlVideo` held by value; the view through `unique_ptr<void, DestroyView>` | 80787a5 |
| s2-app | window.cpp:28 | ES.87 | fix | `if (!view)` | 80787a5 |
| s2-app | window.cpp:56 | ES.20 | fix | `SDL_Event event{}` | 80787a5 |
| s2-app | window.cpp:44 | ES.46 | fix | `points()` refuses 0 and > INT_MAX by Error; pixel sizes clamp a negative report to 0 | 80787a5 |
| s2-app | window.h:45 | I.5, I.6 | fix | `SDL_IsMainThread()` checked before `SDL_Init`, Error otherwise. Not testable without a second thread opening a window (needs a display); verified by reading | 80787a5 |
| s2-app | app/options.h:48 | I.5 | fix | `render_size` refuses a scale outside (0, 1] (NaN, inf included); tested | 80787a5 |
| s2-app | app/options.h:40-44; headless/options.h:88-99 | C.2 | fix | comments say these are plain parse output, made and checked by parse() alone | 80787a5 |
| s2-app | mains | ES.25, Con.4, ES.28 | fix | `const std::optional<SceneDescription> loaded = ...` | 80787a5 |
| s2-app | mains | ES.12 | fix | `loaded`, `app_clock` | 80787a5 |
| s2-app | headless/main.cpp:62 | ES.22 | fix | `render_samples(index)` returns the last sequence; a plan with no sample would be a logic_error (unreachable: the core's plan has at least one; not testable without breaking the core's contract) | 80787a5 |
| s2-app | mains | ES.48 | fix | `(void)` casts dropped (not `[[nodiscard]]`) | 80787a5 |
| s2-app | mains; options | ES.64, ES.48, ES.23 | fix | braces | 80787a5 |
| s2-app | app/main.cpp:88-91; headless/main.cpp:78-79 | ES.77 | fix | `for (events = poll(); !events.quit; events = poll())`; `if (written(...))` instead of `continue` | 80787a5 |
| s2-app | options.cpp ×2 | ES.40 | fix | `number()`/`parsed<T>()` plus a range test | 80787a5 |
| s2-app | options.cpp ×2 | SL.str.2 | fix | `constexpr std::string_view usage` | 80787a5 |
| s2-app | mains, window.cpp, options | SF.10 | fix | includes added | 80787a5 |
| s2-app | headers; generated header | SF.8 | convention | - | - |
| s2-app | all project includes | SF.12 | convention | - | - |
| s2-app | embed_metallib_script.cmake:31 | SL.con.1, SL.str.5 | fix | `std::array<unsigned char, N>` plus a `span<const std::byte>` view | 23ba080 |
| s2-app | MetalLibrary.cmake:20-27, 56-61 | P.3, 4.1 | fix | `-fmetal-math-mode=fast -O2` named; the metallib is byte-identical with them (checked; `-O3` differs, so `-O2` is the default) | 23ba080 |
| s2-app | CMakePresets.json; no IPO | GEN.4 | reject | the frame is GPU-bound (27.7 ms GPU time; the host records a frame in microseconds); LTO would change the build configuration (CLAUDE.md 6.5) with no measured gain to justify it (Per.1, Per.6) | - |
| s2-app | presets; MetalLibrary.cmake | GPU.10 | reject | the GPU counters were captured without such a preset (docs/research/2026-10-10-path-kernel-counters.md); shader line tables change the pinned shader flags, a locked decision (MEMORY.md) that needs its own change | - |
| s2-app | CMakeLists.txt:43 | P.12 | fix (deferred to merge) | `-Wconversion -Wsign-conversion -Wshadow -Wold-style-cast` must build the whole tree clean, which touches src/core and tests: cross-area. Sanitizer preset likewise left to the lead | - |
| s2-app | CMakeLists.txt:20-23 | 4.1, P.12 | fix | `find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)` and `${Python3_EXECUTABLE}` | 23ba080 |
| s2-app | check_boundaries.sh:31,39,47,59 | P.7, 3.1 | fix | grep status >= 2 stops the check (status 2), directories checked first; probe with an unreadable file | 23ba080 |
| s2-app | check_boundaries.sh:20 | P.12 | fix | `@import` caught; probe | 23ba080 |
| s2-app | check_toolchain.py:58 | I.2 | fix | command table built in `main()` from the parsed tools | 23ba080 |
| s2-app | app/main.cpp:109-110 | TLM.6 | fix | title says "(live window, not a benchmark)"; main's header says recorded figures come from an idle GPU | 80787a5 |

## Cross-area

- Core agent: SL.con.4 / COPY.6 `static_assert(std::is_trivially_copyable_v<...>)` in the contract headers (FrameConstants, CameraData, Transform, Bounds, the scene records); `core/contracts/frame_constants.h:43` and `core/frame/history.h:19` cite `metal/frame/accumulation.h` for the 2^24 - 1 rule, which the core holds (`frame::max_accumulated_frames`, history.h); `plan_headless(first, samples, bool, bool, bool)` (core/frame/history.h:124) has adjacent swappable parameters (I.24).
- Shader agent: `src/metal/scene/scene_block.metal.h:52` is longer than 120 columns. My edits to `src/metal/passes/*/*.metal` are only the kernels' parameter lists and one `#include "metal/passes/bindings.h"` line each (needed for them); merge with care.
- Tests agent: an allocation-counting test around `Renderer::record()` and `SceneAcceleration::update()` for the MEM.9 claims "nothing is allocated" (s2-metal-host row MEM.9); `tests/gpu/argument_table_test.cpp`'s header says Apple's documentation leaves the rebinding question unsaid, but MTL4ComputeCommandEncoder.h states it (Metal snapshots the table at each dispatch).
- Lead: `docs/process/MEMORY.md`'s toolchain lock names Xcode, the SDK, the Metal compiler and Apple clang; `cmake/toolchain.json` now also pins the C and Objective-C compilers and CMake, and the SDK checked is the one the build compiles against (the presets set `CMAKE_OSX_SYSROOT`). The headless renderer now refuses a non-empty `--out`, a user-visible behavior change worth a line in QUEUE/MEMORY if the lead keeps such notes.
- Lead (after merging): add `-Wconversion -Wsign-conversion -Wshadow -Wold-style-cast` once the whole tree builds clean under them; a sanitizer preset (s2-app P.12).
- Test files I edited only to keep them compiling: `tests/options_test.cpp` (OptionsError rename, `written()` struct), `tests/gpu/library_test.cpp` (std::byte), `tests/gpu/tone_map_test.cpp` (`FrameImages::Bloom`), `tests/gpu/acceleration_test.cpp` (non-const update/transforms). New test files: `tests/gpu/lifetime_test.cpp`, `tests/gpu/frame_errors_test.cpp`, `tests/programs_test.cpp`; probes added to `tests/test_check_boundaries.sh` and `tests/test_check_toolchain.sh`.

## Measurements

Apple M3 Max, Xcode 26.2, release build. A scratch harness (not committed)
linked against `libserenity_metal.a`: the path tracer (`graphs/path.toml`) on
`scenes/marbles.toml` at 3456 x 2234, 200 frames recorded with two in flight
as the window records them, GPU time per frame from commit feedback, median
of frames 21-199; wall time over all 200; load time = scene read, device,
submission, offscreen target and renderer construction. Each run only on an
idle GPU (utilization 0%, no other serenity or scratchpad GPU program before
or during it; other agents' tests and timing runs share this GPU, and runs
they disturbed were discarded and retried). "Before" is c0bc7d1 (marbles),
"after" is dd8cacc (every change but the comment-only 34b244a), interleaved:

| Round | before: frame median | after: frame median | before: load | after: load |
|---|---|---|---|---|
| 1 | 27.75 ms | 27.69 ms | 383 ms | 381 ms |
| 2 | 27.37 ms | 27.23 ms | 367 ms | 371 ms |
| 3 | 27.40 ms | 27.24 ms | 369 ms | 369 ms |
| 4 | 27.17 ms | 27.18 ms | 372 ms | 367 ms |

No change in frame time or load time beyond run-to-run noise (none is
claimed: the changes were not made for speed). An earlier interleaved set
(before 27.44-27.69 ms, after A-C 27.65-27.70 ms) agrees.

The barrier between frames (frame_images.h's unmeasured claim), same harness,
4 interleaved rounds: with it 27.65-27.70 ms GPU / 27.79-27.85 ms wall a
frame; without it 66.6-78.9 ms / 44.7-48.3 ms. Recorded in
docs/research/2026-10-09-pass-costs.md.

The feedback poll: 297, 307, 314 sleeps over 200 settles in three runs, every
settle sleeping; those runs' frames 27.18-27.25 ms GPU, 27.51-27.62 ms wall,
the same as without the counters.

Metal flags: the metallib built with `-fmetal-math-mode=fast -O2` is byte for
byte the one built without them (`-O3` differs). The binding header changed
no AIR byte (each shader compiled before and after at the same path).

## Tests

- `cmake --build --preset native-release` and `native-debug`: clean under -Werror.
- `./build/native-release/tests/serenity_tests`: 97 cases, 2544 assertions, pass.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/native-release/tests/gpu/serenity_gpu_tests`: 85 cases, pass.
- `ctest --preset native-release` and `ctest --preset native-debug`: 5/5 pass (serenity_tests, boundaries, boundaries_fire, toolchain_fires, serenity_gpu_tests).
- `make check`: pass.

## Counts

126 rows: fix 110 (of them 2 fix (cross-area), 1 fix (deferred to merge)); split fix/reject 5 (a comment or part fixed, the rest rejected with its reason); reject 8; convention 3.
