## Scope and method

- **Files read in full:** `src/app/{clock,options,window}.{h,cpp}`, `src/app/main.cpp`, `src/headless/{options.h,options.cpp,main.cpp}`, `tools/check_boundaries.sh`, `tools/check_toolchain.py`, `CMakeLists.txt`, `CMakePresets.json`, `cmake/MetalLibrary.cmake`, `cmake/embed_metallib_script.cmake`, `cmake/toolchain.json`. There is no `src/CMakeLists.txt`.
- **Build flags:** checked against the real compile lines in `build/native-{debug,release}/compile_commands.json`. Release compiles with `-O3 -DNDEBUG -arch arm64`, no `-g`, no `-flto`. Debug compiles with `-g` and no `-O`.
- **Dependency pins:** checked against the fetched sources. stb_image_write is v1.16, toml++ is at v3.4.0 (`30172438`), metal-cpp is at `f567ed8` (tag `release/metal-cpp_macOS26_iOS26`), SDL is 3.4.18. All four match their comments.
- **Already known:** everything in `sweep1-findings.md` is left out. Where a finding sits next to a known one, the row says so.
- **Corpus servers:** both were up. The C++ Core Guidelines MCP answered, and so did the perf corpus at :7015 through `perf.py`. The separate `cpp-perf-guidelines` MCP server failed to connect (ECONNREFUSED), but `perf.py` reached the same corpus.

## Findings

| file:line | rule ID | rule title | violation | fix | severity |
|---|---|---|---|---|---|
| CMakeLists.txt:41-42 vs 73-78 | P.12 / house rule 4.1 (no ambient behaviour) | Use supporting tools as appropriate | The comment says "Third-party headers are included as SYSTEM, so their warnings are theirs". SDL3's are not: `compile_commands.json` shows `-I…/_deps/sdl3-src/include` and `-I…/sdl3-build/include-revision` for `serenity` (toml++, stb and metal-cpp do get `-isystem`). A new SDL header warning under `-Wall -Wextra -Wpedantic -Werror` would fail the build. | Add `SYSTEM` to `FetchContent_Declare(SDL3 …)` (CMake 3.25, already the minimum), or set `SYSTEM` on the target. | medium |
| src/app/window.h:28-29 vs :44-46; src/app/main.cpp:75 | I.4 | Make interfaces precisely and strongly typed | `frame::Extent` is documented as "the size of an image in pixels" (`core/frame/extent.h:9`). `Window(const char*, frame::Extent size)` takes it as **points** ("`size` points across"). Lines 28-29 of the same header say "Sizes are in pixels, not points". So 1280x800 opens a 2560x1600-pixel window. The type and the comments contradict each other. | Give the constructor its own `Points{w,h}` type, or take pixels. Fix the header comment either way. | medium |
| src/headless/options.h:68 | NL.2 (stale cross-reference) | State intent in comments | It cites "2^24 - 1, metal/frame/accumulation.h", but `accumulation.h` holds no such limit. It is `frame::max_accumulated_frames` at `core/frame/frame_inputs.h:31`, and the rule is in `core/frame/history.h:33`. | Cite `core/frame/frame_inputs.h` (`max_accumulated_frames`). | low |
| src/headless/options.h:45 | NL.2 | State intent in comments | It cites `metal/frame/accumulation.h` for "frames at two instants are two scenes". `accumulation.h` says the core decides this (`core/frame/history.h`) and does not state the rule itself. | Cite `core/frame/history.h`. | low |
| src/app/window.h:15 | NL.2 | State intent in comments | "the only code that uses SDL". `src/app/clock.cpp:3` also includes `<SDL3/SDL_timer.h>` and calls `SDL_GetTicksNS`. | Correct the comment, or move the clock off SDL (next row). | low |
| src/app/clock.cpp:3,7,10-11 | SL.2 / ES.1 | Prefer the standard library to other libraries | The monotonic clock is SDL's tick counter. `std::chrono::steady_clock` gives the same guarantee and needs no SDL. As written, the clock depends on SDL being initialised and ties the "one clock" to the window library. | Use `std::chrono::steady_clock::time_point start_`, and return `frame::Seconds{steady_clock::now() - start_}`. | low |
| src/app/clock.cpp:11 | ES.45, ES.48, P.1 | Avoid magic constants / Avoid casts / Express ideas directly | The unit conversion is hand-written: `static_cast<double>(now - start_ns_) * 1e-9`. | `frame::Seconds{std::chrono::nanoseconds{now - start_ns_}}`: no cast, no literal. | style |
| src/app/clock.h:19-20 | NL.2 (TLM.11 citation imprecise) | State intent in comments | "GPU timestamps run on a different clock". The timestamps this program uses are already host time (`submission.h:79`, `frame_times.h:18`: "Both instants are host time, on one clock"). What differs is the origin (SDL init vs host boot), not the clock domain. TLM.11 is about raw GPU counters. | Say the GPU instants are host time with a different origin, so the two are never subtracted. | low |
| src/app/options.h:28-29 | NL.2 | State intent in comments | "nothing measured goes through it (measurement is headless …)". The same program measures GPU time and shows it in its title (`main.cpp:13-18`, 109-110). | Narrow the claim: the stretch is not part of the GPU time shown. | style |
| src/headless/options.cpp:17-18 | NL.2 / NL.3 | Keep comments crisp | The comment for `seconds()` mentions a parameter `least` that does not exist. The parameters are `(option, text, positive)`. | "Seconds from `text`: finite, and > 0 when `positive`, else >= 0." | style |
| src/app/window.h:70, window.cpp:21 | NL.2 | State intent in comments | The comment says "SDL_Init and SDL_Quit for the video subsystem". `SDL_Quit()` shuts down every subsystem, not just video. | Use `SDL_QuitSubSystem(SDL_INIT_VIDEO)` to match the pairing, or reword the comment. | low |
| src/app/window.h:34-35 | Per.6 | Don't make claims about performance without measurements | "costs nothing measurable against a frame" is a performance claim with no measurement cited. | Cite a measurement, or drop the claim. | style |
| src/app/main.cpp:68; src/headless/main.cpp:36 | ES.42, ES.100 | Keep use of pointers simple / Don't mix signed and unsigned | `{argv + 1, static_cast<std::size_t>(argc - 1)}`. POSIX allows `argc == 0` (execve with an empty argv). Then `argc - 1` becomes `SIZE_MAX` and `argv + 1` points past the array, which is UB. | `std::span(argv, argc).subspan(argc > 0 ? 1 : 0)`, or check `argc < 1` and fail. | low |
| src/app/options.cpp:37-40; src/headless/options.cpp:20-24, 37-40 | E.28 | Avoid error handling based on global state (errno) | All three parsers detect errors through `errno` (`strtod`/`strtoull`). The integer path also needs a hand-written guard against signs and whitespace (`options.cpp:34`). | Integers: `std::from_chars`, which has no errno or locale and rejects signs and spaces by design. Doubles: `from_chars` for floating point may not be in this libc++ (AppleClang 17); if not, keep `strtod` and note why. | low |
| src/app/options.cpp:26,31,36; src/headless/options.cpp:62-67 | ES.86 | Avoid modifying loop control variables inside the body of raw for-loops | `args[++i]` inside `for (…; ++i)`. In headless it is hidden inside the `value` lambda, which captures `i` by reference. | Walk a cursor with a `next()` that returns `optional<const char*>`, or use a `while` loop with explicit consumption. | style |
| src/headless/options.h:102, options.cpp:139 | I.24 | Avoid adjacent parameters that can be swapped | `written(Write, std::uint64_t n, std::uint64_t frames)`: `written(w, frames, n)` compiles and is wrong. | Pass a struct, or strong types (`FrameOffset`, `FrameCount`). | low |
| src/headless/options.cpp:46 | I.24 | (same) | `side(string_view option, string_view text, string_view whole)` has three adjacent `string_view`s. | Reorder, or take a small struct. | style |
| src/headless/main.cpp:54-55 (declared at core/frame/history.h:124) | I.24 | (same) | At the call site `plan_headless(first, samples, bool, bool, bool)`, two `uint64_t`s and three `bool`s are adjacent. The declaration is outside this scope. | Pass a named struct of flags. | low |
| src/headless/options.cpp:149-150 | P.6 / house rule 3.1 | What cannot be checked at compile time should be checkable at run time | After the exhaustive `switch`, `return true;` silently treats an out-of-range `Write` as "write". That is a success-shaped fallback. | `throw std::logic_error` (or `__builtin_unreachable()`), not `return true`. | low |
| src/app/main.cpp:52-56 | ES.27, SL.con.1, SL.io.3, SL.4 | Use std::array / Prefer STL array / Prefer iostreams / Type-safe use | `char text[200]` plus `std::snprintf`. The return value is ignored, so text over 200 characters is cut off silently. The `%u`/`%llu` specifiers plus a cast stand in for type safety. | `std::format("Serenity  |  {} x {}  |  GPU {:.2f} ms …", …)` (libc++ 17 has `<format>`). | low |
| src/headless/main.cpp:83-84 | ES.27, SL.con.1, SL.io.3, ES.45 | (same) | `char name[32]` plus `std::snprintf("frame-%06llu.png")`, with a cast and the magic number 32. | `std::format("frame-{:06}.png", index)`. | style |
| src/app/main.cpp:116; src/headless/main.cpp:87,92 | SL.io.3 | Prefer iostreams for I/O | `std::fprintf`/`std::printf` (not type-safe). | `std::cerr << …`, or `std::format` plus `fputs`. | style |
| src/app/main.cpp:52,53-55 (1e3 x3),75 (1280, 800, "Serenity"),84 (1.0); src/headless/main.cpp:49 (4),83 (32); src/headless/options.h:91,93 (1/60, 1920x1080) | ES.45 | Avoid magic constants | Unnamed literals. For ms, `std::chrono::duration<double, std::milli>{summary.mean}.count()` needs no constant. The headless 4 bytes per RGBA pixel repeats `metal/device/offscreen.cpp:25`. | Named `constexpr` constants (`initial_window_points`, `title_period`, `bytes_per_rgba8`), or chrono conversions. | style |
| src/headless/main.cpp:49 vs metal/device/offscreen.cpp:25-26 | ES.3 | Don't repeat yourself | The readback buffer size (`w*h*4`) is computed in both places. The two must agree or `read_rgba` throws. | Have `Offscreen` expose `rgba_bytes()`, or allocate the buffer itself. | low |
| src/app/options.cpp:36-43 vs src/headless/options.cpp:19-30; app/options.h:35, headless/options.h:77, app/window.h:37 | ES.3 | Don't repeat yourself | The same `strtod` validation is written twice. Three identical `runtime_error` subclasses exist. | Share one number parser. Keep separate error types only where they mean different axes, and say so. | style |
| src/app/options.h:37; src/headless/options.h:79; src/app/window.h:39 | C.52 | Use inheriting constructors | Each error class writes out a forwarding constructor. | `using std::runtime_error::runtime_error;` | style |
| src/app/options.h vs src/headless/options.h | NL.8 | Use a consistent naming style | The same role has two names: `app::OptionsError` and `headless::Error`. | Pick one name. | style |
| src/app/window.cpp:43-49 | C.49 | Prefer initialization to assignment in constructors | `window_.reset(...)` and `view_ = make_unique(...)` are assigned in the constructor body. | Initialise them in the member-init list through helpers (`make_window(title,size)`), which throw on failure. | style |
| src/app/window.h:48-51 | COPY.5 | Express unique-ownership resources as move-only types | Moves are deleted. The stated reason ("SDL's video subsystem is one per process") is a reason against copying, not moving. All three members are `unique_ptr`s, so `= default` noexcept moves are safe. | Default the moves (noexcept), or state the real reason they are not allowed (for example, a layer pointer handed out). | style |
| src/app/window.cpp:15-24, 26-37 | C.21 | Define or delete all of copy, move, destructor | `SdlVideo` and `MetalView` declare a destructor and deleted copy, but not move. Moves are suppressed only implicitly. | Delete the moves explicitly. | style |
| src/app/window.h:77-79, window.cpp:43,49 | R.5 | Prefer scoped objects, don't heap-allocate unnecessarily | `SdlVideo` is empty, and `MetalView` wraps a `void*` (`SDL_MetalView`). Both are heap-allocated only to stay incomplete in the header. | Use `std::unique_ptr<void, DestroyView>`, as `window_` already does, and a complete empty `SdlVideo` with out-of-line constructor and destructor. | style |
| src/app/window.cpp:28 | ES.87 | Don't add redundant == or != to conditions | `if (view == nullptr)` | `if (!view)` | style |
| src/app/window.cpp:56 | ES.20 | Always initialize an object | `SDL_Event event;` is left uninitialised. | `SDL_Event event{};` | style |
| src/app/window.cpp:44 | ES.46 | Avoid lossy arithmetic conversions | `static_cast<int>(size.width)` is unchecked: a `uint32_t` above `INT_MAX` becomes negative. | Check the range, or use `gsl::narrow`. | style |
| src/app/window.h:45 | I.5 / I.6 | State preconditions / Prefer Expects() | "Must be called on the main thread" is stated but not checked. SDL3 has `SDL_IsMainThread()`. | Check it and throw `Error`. | low |
| src/app/options.h:48, options.cpp:54-59 | I.5 | State preconditions | `render_size(window, scale)` states no domain for `scale`. Above about `2^32/pixels`, `static_cast<uint32_t>(double)` is UB. NaN is silently clamped to 1 by `std::max`. | State that scale is in (0, 1] and check it. | low |
| src/app/options.h:40-44; src/headless/options.h:88-99 | C.2 | Use class if the class has an invariant | `Options` carries invariants (scale in (0,1]; frames ≥ 1; samples ≤ max; first+frames bound) that are enforced only by `parse()`. The type does not hold them. | Accept as plain parse output, but say so. Or make the fields private behind `parse()`. | style |
| src/app/main.cpp:70-73; src/headless/main.cpp:38-41 | ES.25 / Con.4 / ES.28 | Declare const unless modified / lambda for complex init | `std::optional<SceneDescription> scene` is left mutable only so it can be assigned once. | `const auto scene = options.scene.empty() ? std::optional<…>{} : std::optional{scene::load(options.scene)};` | style |
| src/app/main.cpp:70,72,87,101; src/headless/main.cpp:38,40,55,69 | ES.12 | Do not reuse names in nested scopes | The local `scene` hides namespace `serenity::scene`, which `using namespace serenity` brings in. `scene = scene::load(...)` reads ambiguously. In `app/main.cpp:83`, `clock` hides `::clock`. | Rename the local to `loaded_scene` and the clock to `app_clock`. | style |
| src/headless/main.cpp:62 | ES.22 | Don't declare a variable until you have a value | `std::uint64_t sequence = 0;` is a placeholder that looks like a real sequence. It is correct only because `plan.samples ≥ 1`, which is an unstated reliance on `plan_headless`. | Return the last sequence from a helper that renders the samples, or assert `plan.samples ≥ 1`. | style |
| src/app/main.cpp:113; src/headless/main.cpp:73,89 | ES.48 | Avoid casts ("Flag all C-style casts, including to void") | `(void)submission.finish();` and `(void)submission.wait_until_complete(…)`. Neither function is `[[nodiscard]]`, so the casts are noise. If they were, the guideline would require `std::ignore =`. | Drop the casts, or mark the functions `[[nodiscard]]` and use `std::ignore =`. | style |
| src/app/main.cpp:84,101; src/headless/options.cpp:89,92; src/app/options.cpp:41,49 | ES.64 / ES.48 / ES.23 | Use T{e} notation | Functional-style casts and constructions: `frame::Seconds(1.0)`, `std::string(arg)`, `std::optional(scene->camera)`. Parenthesised object construction throughout both mains. | `frame::Seconds{1.0}`, `std::string{arg}`, braced construction. | style |
| src/app/main.cpp:88-91; src/headless/main.cpp:78-79 | ES.77 | Minimize break and continue | `for (;;) { … break; }` and `continue`. | `while (!(events = window.poll()).quit)`, or a helper. Invert the `written` test. | style |
| src/app/options.cpp:40; src/headless/options.cpp:24 | ES.40 | Avoid complicated expressions | A six-clause validity condition. | A named helper `parsed_fully(…)` plus a range test. | style |
| src/app/options.cpp:14; src/headless/options.cpp:13 | SL.str.2 | Use std::string_view to refer to character sequences | `constexpr const char* usage`. | `constexpr std::string_view usage`. | style |
| src/app/main.cpp:31-46; src/app/window.cpp:84; src/app/options.cpp:20; src/headless/main.cpp:16-31,54,65; src/headless/options.cpp:60 | SF.10 | Avoid dependencies on implicitly #included names | Names used without their header, beyond the known `<cstdint>`/`<filesystem>`: `frame::FrameInputs`/`frame::Seconds` (`core/frame/frame_inputs.h`) in both mains, `frame::Extent` (`core/frame/extent.h`) in `app/main.cpp`, `frame::accumulates` (`core/frame/schedule.h`) in headless main, `std::size_t` (`<cstddef>`) in all four .cpp files, `std::uint32_t` in `window.cpp`. | Include what you use. | style |
| all four in-scope headers; generated `serenity/metallib/<NAME>.h` (embed_metallib_script.cmake:26) | SF.8 / P.2 | Use #include guards / Write ISO Standard C++ | `#pragma once` is a non-ISO extension. (This is a project-wide convention.) | Guards, or accept and record the convention. | style |
| all project includes (e.g. app/main.cpp:36-46) | SF.12 | Quoted form only for files relative to the includer | `"core/frame/…"`, `"metal/…"` are found through the `-I src` search path, not relative to the including file. The guideline asks for `<…>`. (Project-wide convention.) | `<core/frame/graph_file.h>`, or record the deviation. | style |
| cmake/embed_metallib_script.cmake:31 | SL.con.1 / SL.str.5 | Prefer std::array / Use std::byte for non-character bytes | `inline constexpr unsigned char NAME[]` is a C array of raw bytes. | `inline constexpr std::array<std::byte, ${size}>`, or at least `std::array<unsigned char, N>`. | style |
| cmake/MetalLibrary.cmake:20-27, 56-61 | P.3 (house rule 4.1) | Express intent | The header says flags are pinned "explicit rather than the compiler's default". The math mode ("Math is Metal's default (fast)") and the optimisation level are still left to the compiler's defaults. The Metal compiler-version pin limits the risk. | Pass `-fmetal-math-mode=fast` and the `-O` level explicitly. (This is distinct from the known INFINITY-under-fast-math shader finding.) | low |
| CMakePresets.json:29-37; CMakeLists.txt (no IPO) | GEN.4 | Enable ThinLTO as the default | `native-release` (the build `make run/headless/movie` use) links `serenity_core`/`serenity_metal` static archives into the executables without LTO. Host CPU work per frame is small, so the expected gain is small. | `check_ipo_supported` plus `INTERPROCEDURAL_OPTIMIZATION ON` for Release, or record why not. | low |
| CMakePresets.json; cmake/MetalLibrary.cmake | GPU.10 (gpu category; supplementary) | Profile with GPU timelines and counters | The build used for measurement (Release) has no `-g`. No preset carries symbols plus `-O3`, and no shader build has `-gline-tables-only`/`-frecord-sources`. So per-line GPU counter attribution and CPU symbolication in Instruments are unavailable without hand-editing flags. | Add a `native-profile` preset (RelWithDebInfo-style `-O3 -g`, `-DNDEBUG`) plus shader line tables. | low |
| CMakeLists.txt:43 | P.12 (enforcement of ES.46, ES.48/49, ES.12) | Use supporting tools as appropriate | Warnings omit `-Wconversion -Wsign-conversion -Wshadow -Wold-style-cast`. With them, the narrowing, C-style casts and shadowing above would fail the build. There is no sanitizer preset and no `.clang-tidy`. | Add the flags, plus an ASan/UBSan debug preset. | low |
| CMakeLists.txt:20-23 | house rule 4.1 / P.12 | (environmental determinism) | The configure-time check runs `python3` from PATH. The interpreter is ambient and unpinned. The known "cmake not pinned" finding does not cover it. | `find_package(Python3 REQUIRED COMPONENTS Interpreter)` and `${Python3_EXECUTABLE}`, with a version floor. | low |
| tools/check_boundaries.sh:31,39,47,59 | P.7 / house rule 3.1 | Catch run-time errors early | `grep … 2>/dev/null \|\| true` throws away grep's exit status 2 (missing directory, unreadable file). If `src/core/` is renamed, every rule reports nothing and the script prints "boundaries OK". | Fail when grep exits with status ≥2. Check that the scanned directories exist first. | low |
| tools/check_boundaries.sh:20 | P.12 | Use supporting tools as appropriate | The include pattern catches only `#include`/`#import <Fw/…>`. The Objective-C module syntax `@import Metal;` is not caught (separate from the known missing framework names). | Add `@import[[:space:]]+(Metal\|…)`. | low |
| tools/check_toolchain.py:58 | I.2 (by analogy) | Avoid non-const global variables | `--cxx` mutates the module-global `COMMANDS` dict. | Build the command table inside `main()` from the parsed arguments. | style |
| src/app/main.cpp:109-110 | TLM.6 | Diagnostic mode is not benchmark mode | The title prints "GPU x ms" with the resolution only. `frame_times.h:25-26` says whoever shows a frame time names "this machine and this build", and the build is not named. A window run is also not an idle-GPU measurement (user memory: a live window doubles frame times). | Label the title "(window, not a benchmark)" and the build type (`NDEBUG`), or drop the timings from the title. | style |

## Guideline citations in comments, verified

| Location | Citation | Verdict |
|---|---|---|
| app/clock.h:14 | I.1 | Correct (implicit inputs) |
| app/clock.h:20 | TLM.11 | Rule exists and is relevant. The surrounding claim is imprecise (see row above). |
| app/options.h:33, headless/options.h:66 | E.2, E.14 | Correct |
| app/window.h:20 | R.1, R.20 | Correct |
| app/window.h:31 | E.2, E.5, E.14 | Correct |
| headless/main.cpp:12 | GPU.1 | Correct. GPU.1 treats readback as a step boundary. |

Other cross-references:
- **Correct:** `metal/device/presenter.h` (app/main.cpp:11), `metal/frame/renderer.h` (app/options.h:21), `metal/film/non_finite.h`, and the `docs/architecture/{file-mapping,change-axes,logical-overview}.md` files all exist and say what is cited.
- **Wrong:** `headless/options.h:45` and `:68` (rows above).

## Categories walked

**Core Guidelines:** P (13), I (20), F (40), C (100), Enum (8), R (25), ES (66), Per (18), E (22), Con (5), SF (16), SL (22), NL (20).

**Perf corpus:** codegen (8), memory (11), copy-move (9), lifetime (8), concurrency (8), telemetry (11).

## Rules judged with no violation

- **P:** P.1–P.11 and P.13. P.2, P.3, P.6, P.7 and P.12 are cited above.
- **I:** I.1, I.2 (C++), I.3, I.7–I.13, I.22, I.23, I.25–I.27, I.9.
  - I.4 for `Window::metal_layer()` returning `void*`: not a violation, because it is wrapped in `metal::LayerHandle` at once (I.30) and the header documents it.
  - I.30 holds for the same reason.
- **F:** all except those cited above.
  - Parameter passing (F.15–F.21, F.25) is correct throughout: `Extent` by value, `const std::string&`, `span` for args, zstrings for SDL.
  - F.8: `render_size` and `written` are pure.
  - F.46: both mains return int.
  - F.52: lambdas capture locally.
  - F.55: not applicable (no function defines varargs).
- **C:** everything else, including C.12, C.13 (known), C.31, C.35–C.37, C.41, C.42, C.46, C.47, C.80, C.81 and C.131.
- **Enum:** Enum.1–Enum.8. `Write` is an enum class with lowercase enumerators, no underlying type and no explicit values.
- **R:** all except R.5. R.1, R.3, R.11, R.20 and R.23 hold.
- **ES:** everything else, including ES.5, ES.6, ES.10, ES.11, ES.26, ES.31–ES.34, ES.47, ES.49, ES.50, ES.56, ES.60–ES.65, ES.70–ES.76, ES.78, ES.79, ES.101 and ES.102.
  - ES.101: `((n + 1) & n)` is unsigned bit manipulation, as it should be.
  - ES.103 at `headless/main.cpp:49`: `w*h*4` can overflow only for sides around 2^32. `Offscreen` throws first, because Metal makes no texture above 16384, so it is unreachable. This depends on the device.
  - ES.46 at `headless/main.cpp:61`: `index`→double loses precision above 2^53. This is accepted as unreachable in practice.
- **Per:** Per.1–Per.19 and Per.30 hold. There is no allocation per frame in either loop beyond the once-a-second title string.
- **E:** everything else.
  - E.14 at `headless/main.cpp:75`: `throw std::runtime_error` is allowed by E.14's own example ("good" for generic errors).
  - E.15, E.17, E.18, E.31: each main has one top-level `catch (const std::exception&)` and returns 1.
  - The lifetime-on-throw issue in the mains is already known (C.13/GPU.7).
- **Con:** Con.1–Con.5 hold, apart from the ES.25 rows. Member functions are const where they can be.
- **SF:** SF.1–SF.7, SF.9, SF.11, SF.13, SF.20–SF.22. Helpers are in anonymous namespaces, and `using namespace` appears only at local scope.
- **SL:** SL.1, SL.3, SL.C.1, SL.con.2–SL.con.4, SL.io.1, SL.io.2 (in scope), SL.io.50, SL.str.1, SL.str.3, SL.str.4, SL.str.10–SL.str.12. SL.io.10 does not apply because printf is used.
- **NL:** NL.4, NL.5, NL.7, NL.9–NL.11, NL.15–NL.21, NL.25–NL.27, NL.1, NL.3. Naming is consistent CamelCase for types and snake_case otherwise.
- **codegen:**
  - GEN.1–GEN.3, GEN.6, GEN.7: no hints or attributes are used, and no host loop is hot.
  - GEN.5: PGO is not feasible or worthwhile, because host CPU is not the bottleneck.
  - GEN.8: no reliance on vectorisation or volatile.
  - `-ffast-math` is not used for C++.
  - `NDEBUG` is set in Release, and no `assert` appears in scope.
  - `-O3` in Release.
- **memory:** MEM.1–MEM.11 hold. Only one buffer is allocated, at start-up and outside any loop. MEM.6 and MEM.9 target engine subsystems and embedded code, not a leaf `main`.
- **copy-move:** COPY.1–COPY.4 and COPY.6–COPY.9 hold.
  - COPY.7: the per-frame `std::optional(scene->camera)` copy is needed, because `FrameInputs` holds the camera by value.
- **lifetime:** LIFE.1–LIFE.8 hold. There are no statics and no raw storage.
- **concurrency:** CONC.1–CONC.8 do not apply. Both programs are single-threaded on the host.
- **telemetry:**
  - TLM.1: `FrameTimes` is a displayed feature fed by commit feedback, not a probe.
  - TLM.2–TLM.5, TLM.7, TLM.8, TLM.10: no instrumentation layer exists.
  - TLM.9: the clock source is named and monotonic.
  - TLM.11: no clock domains are mixed. `FrameTimes` subtracts host-time pairs only, and paces itself on the CPU clock.