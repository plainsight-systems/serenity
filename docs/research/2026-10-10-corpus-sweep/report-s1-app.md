## Review: src/app, src/headless, build, tools, scripts (read-only)

I checked both guideline sources: the C++ Core Guidelines over MCP, and the performance corpus over HTTP on :7015, which worked. The cpp-perf MCP server itself was down.

### Findings

**1. Medium: on an exception, GPU resources are freed while frames are still in flight**
- **Where:** `src/app/main.cpp:76-80`, `src/headless/main.cpp:43-46`
- **Rules:** R.1 / E.6 (cleanup on every path goes through RAII); C.13 (whatever another object depends on must be declared before it, so it is destroyed after it); GPU.7 (wait on the event that guards a resource's lifetime).
- **What is wrong:**
  - `~Submission` waits for committed work, but it is declared before `Renderer`, `Presenter` and `Offscreen`, so it is destroyed after them. Those three have `= default` destructors.
  - When an exception leaves the loop mid-run, their textures and buffers are released while submissions may still be using them. Metal 4 command buffers do not keep the resources they use alive.
  - Examples of such exceptions: `begin()` reporting a failure in frame n while n+1 is still running, a history `Error` from `render_to_*`, or `set_title`.
  - The repo already treats this as unsafe elsewhere: `accumulation.cpp:24-29` and `frame_images.cpp:50-64` call `drain()` before `release_resident`, with the comment "Frames in flight may still read the old image."
  - Only the normal path is safe, because it calls `finish()` first.
- **Fix:** add a guard declared after the renderer and its target, whose noexcept destructor does the same bounded wait as `~Submission`. A `Submission::quiesce() noexcept` that both use would serve. Alternatively, catch inside the try block, wait, and rethrow.

**2. Medium: the boundary check is not run on the default verification path**
- **Where:** `CMakeLists.txt:105`, `Makefile:14-23, 80-86`; `tests/CMakeLists.txt` has no `add_test` for it.
- **Rules:** MEMORY.md locked decision ("Enforced by `tools/check_boundaries.sh`"); engineering_philosophies "Test the Contract" (non-trivial changes need deterministic verification); cpp_architecture_review Dependency Checks.
- **What is wrong:** only `make check` runs the script. `make test`, `make test-release` and `ctest` never do, and there is no CI (`Makefile:5`). A core include of `Metal/` passes every build and test.
- **Fix:** `add_test(NAME boundaries COMMAND sh ${CMAKE_SOURCE_DIR}/tools/check_boundaries.sh)` plus an `add_test` for `tests/test_check_boundaries.sh`. Or make `test` and `test-release` depend on `check`.

**3. Low: the boundary patterns miss part of Metal's host API**
- **Where:** `tools/check_boundaries.sh:20-21`
- **Rule:** MEMORY.md ("only `src/metal/` uses Metal's host API").
- **What is wrong:**
  - `PLATFORM_TOKENS` is `(MTL|NS|CA|MTL4)::`. The pinned metal-cpp also declares `namespace MTLFX` and `namespace MTL4FX` (8 occurrences each in `_deps/metal_cpp-src`), and MEMORY plans to use MetalFX.
  - The include pattern requires `Metal/` straight after the bracket, so `<MetalPerformanceShaders/…>`, `<MetalPerformanceShadersGraph/…>`, `<IOSurface/…>` and `<CoreVideo/…>` get through.
- **Fix:** use `(MTL[A-Z0-9]*|NS|CA)::` and `Metal[A-Za-z]*/|IOSurface|CoreVideo`, and add a case for each to `tests/test_check_boundaries.sh`.

**4. Low: the toolchain pin covers less than the build actually uses**
- **Where:** `CMakeLists.txt:12-26`, `tools/check_toolchain.py:31`, `CMakePresets.json:12-17`
- **Rules:** MEMORY.md toolchain lock ("configuring fails on any mismatch"); CLAUDE.md §4.1 environmental determinism and §4.3 explicit dependency versions.
- **What is wrong:**
  - Only `--cxx` is checked. SDL is built with `CMAKE_C_COMPILER` and `CMAKE_OBJC_COMPILER`, which are never checked. A configure without the preset, or with `-DCMAKE_C_COMPILER=…`, builds SDL with an unpinned compiler.
  - Under the installed CMake (4.2.3), `CMAKE_OSX_SYSROOT` is empty (`build/native-release/CMakeCache.txt`). C++ therefore compiles against whatever SDK `SDKROOT` or xcrun's default resolves to, while the check reads `xcrun --sdk macosx`. A different `SDKROOT` passes the check.
  - CMake itself is not pinned, and its version sets the compile command lines, including whether the sysroot is set at all.
- **Fix:**
  - Call `enable_language(C OBJC)` before the check and pass `--cc` and `--objc` to it.
  - Set `"CMAKE_OSX_SYSROOT": "macosx"` in the base preset, and check that SDK's version.
  - Add a `cmake` pin to `cmake/toolchain.json`.

**5. Low: `make movie` uses a rounded time step**
- **Where:** `Makefile:66`
- **Rule:** MEMORY.md determinism (time is an input; the headless renderer uses a fixed timestep).
- **What is wrong:** `awk 'BEGIN { print 1 / $(FPS) }'` prints 6 significant digits. I ran it and got `0.0166667`, not 1/60. Frame 599 of a movie is rendered at 9.98335 s, but at 9.98333 s from `make headless` (default `1.0/60.0`). For any moving scene, the same frame index gives different images depending on which entry point rendered it.
- **Fix:** add `--fps N` to `headless::parse` (step computed as `1.0/N`), or at least use `awk 'BEGIN { printf "%.17g", 1/$(FPS) }'`.

**6. Low: frames from an earlier run stay in the output directory**
- **Where:** `Makefile:43-46`, `src/headless/main.cpp:48`
- **Rule:** principle 1 as stated in the file headers ("the same command writes the same files").
- **What is wrong:** `create_directories` accepts a directory that already holds frames. Running `make headless FRAMES=30` after `FRAMES=60`, or with `--write last`, leaves the earlier run's `frame-NNNNNN.png` files beside the new ones, and nothing tells them apart. The `movie` target clears its directory (`:64`); `headless` does not.
- **Fix:** refuse a non-empty `--out` with an `Error`, or `rm -rf $(OUT)` in the target.

**7. Low: names reached through transitive includes, and an undeclared target dependency**
- **Rules:** SF.10 (avoid depending on names you did not include yourself); cpp_architecture_review Dependency Checks ("transitive includes or undeclared target dependencies").
- **Where and what:**
  - `src/app/main.cpp:51,81,84` uses `std::uint64_t` without `<cstdint>`.
  - `src/headless/main.cpp:48-49,85` uses `std::filesystem` and `std::uint8_t`/`std::uint64_t` without `<filesystem>` or `<cstdint>`.
  - `CMakeLists.txt:170-172`: `serenity_app_options` includes `core/frame/extent.h` through a raw include directory instead of declaring a dependency on `serenity_core`.
- **Fix:** add the includes. Use `target_link_libraries(serenity_app_options PUBLIC serenity_core)` instead of the bare include directory.

**8. Low: comment anchoring depends on the user's git config**
- **Where:** `scripts/post_commit_review.py:29,33`
- **Rule:** CLAUDE.md §4.1 (no dependence on undeclared machine configuration).
- **What is wrong:** `git show` follows `diff.noprefix` and external diff drivers. The parser assumes `+++ b/`. With `noprefix` set, every path is cut wrong and every finding quietly falls back to the summary comment.
- **Fix:** `git -c core.quotepath=false show --no-ext-diff --src-prefix=a/ --dst-prefix=b/ …`

### Checked and fine
- **Argument parsing:**
  - Both parsers reject unknown options and missing values.
  - `strtod`/`strtoull` results are checked for `errno`, an empty parse, trailing characters and non-finite values.
  - Leading signs and spaces are rejected for counts.
  - Overflow is checked for `first + frames` and for `× samples`.
  - `--size` handles a missing `x` and a zero side.
  - The `frame-%06llu.png` buffer is big enough for `UINT64_MAX` (31 of 32 bytes).
- **Exceptions at `main`:** each `main` catches `std::exception`, prints the program name and the message, and returns 1. `finish()` and `wait_until_complete()` throw on GPU failure, so the `(void)` casts throw away only records, not errors.
- **SDL lifetime:**
  - The `Window` members are released in the right order: view, then window, then `SDL_Quit`.
  - If construction fails partway, the parts already built are cleaned up.
  - Copying and moving are deleted.
  - A null `SDL_Metal_GetLayer` is rejected by `Presenter` (`presenter.cpp:8`).
  - The `Presenter` keeps its own reference to the layer and is destroyed before the window.
- **Determinism:**
  - Headless time is `step × index`, or `--time`, and nothing reads a clock.
  - The only clock reads are in `app/clock.cpp` (SDL's monotonic clock) and the timeouts in `submission.cpp`, which do not affect rendered output.
- **Window loop:** no allocation per frame in `poll()`, `FrameTimes` or `LiveHistory`. The title string is built once a second. The autorelease pool per frame in `render_to_window` is correct. The only blocking is the drawable acquire and the wait for a free slot, which is intended pacing.
- **Warnings:** `-Wall -Wextra -Wpedantic -Werror` is on every first-party target in scope. Third-party headers are included as `SYSTEM`. There are no `assert()` calls, so Release does not quietly drop any checks.
- **Dependencies:** pinned by commit SHA (metal-cpp, toml++, stb) or by URL plus SHA-256 (SDL).
- **Shaders:** compiled at build time with explicit `-std=metal4.0`, `-mmacosx-version-min` and `-Werror`, with depfile tracking. They are embedded as a constexpr array, and a missing or empty metallib fails the build. Metal's default math is the locked decision in MEMORY.md.
- **Toolchain check failure path:** if `python3` cannot run, `RESULT_VARIABLE` is not 0, so configuring fails.
- **`codex-review.sh`:** the preflight branches on curl's exit status, the `|| { RC=$? }` captures codex's exit status, and an empty or non-JSON review is refused.