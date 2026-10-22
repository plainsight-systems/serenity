# Workflow

This repo adapts the Lite Factory workflow from
`governance/product_memory_workflow.md`. The design of each change lives in
the code it changes, not in a separate packet.

## Default Chain

```text
Coordinator
  -> file headers: the contract, the design, and the guidelines behind it
  -> review of the headers
  -> implementation and tests
  -> review
  -> MEMORY.md and QUEUE.md updates
```

## Local Notes

This repo is C++-dominant and performance-sensitive.

- **The headers are the plan.** Each new or changed file's header states its
  contract and its design, and cites by ID the guidelines from the
  `cpp-guidelines` and `cpp-perf-guidelines` corpora that shape it (for
  example R.1, E.5, GDSA.2). The headers are written and reviewed before the
  implementation.
- **Every C++ step is checked against `cpp-guidelines` before its code is
  written, and performance-sensitive work against `cpp-perf-guidelines`.**
  The citations in the headers are the record. If either server is
  unreachable, say so and get explicit agreement before writing C++.
- **Optimizations are designed in, not deferred.** Each carries
  `// Optimization: …` with its concrete reason and the guideline or system
  it follows, so a reader sees a choice rather than an accident. Where it can
  be measured, it is, and the figures go in the note.
- **A data path's cost is derived before it is built.** Its header walks
  every stage and counts, as functions of the resolution, the light count and
  the scene size: the bytes it moves and the copies it makes, the operations
  in its inner loop and whether each can be inlined, and the crossings
  between CPU and GPU in a frame (command buffers, encoders, dispatches,
  synchronizations, readbacks). A count that scales with pixels or lights
  where it could scale with tiles or passes is designed out before code is
  written. Once the path runs it is measured against that floor.
- **Correctness is shown on the GPU** (MEMORY.md): rendering against the
  reference render, sampling against the distribution it should draw from.
- **Independent review is on request,** not per commit:
  `scripts/codex-review.sh <commit>` checks a commit for correctness and
  speed and writes its findings to `.cache/reviews/`. It needs the codex
  CLI; its own test is `make review-check`, apart from `make check`, so a
  developer without it builds and checks everything else.
- A frame's path, acceleration-structure builds, memory footprint and GPU
  dispatch are performance-sensitive by default.
- GPU behavior is environment-sensitive. Verification names the machine it
  ran on (each backend's headless integration test prints it); a green build is not proof. Two
  machines: the development machine, an M3 Max (Metal), and the RTX 3090
  Ti's (CUDA and OptiX, from the backend's Phase 2; MEMORY.md).
- **The RTX 3090 Ti's machine** (`ssh ai`, Ubuntu 24.04;
  `docs/research/2026-10-19-cuda-backend.md`). The same presets and make
  targets as the Mac's: `cmake/toolchain.json`'s "linux" set pins its
  compilers, Ubuntu's clang 18 and CMake 3.28, and LLVM's own release of
  21.1.8 for the lint, unpacked (no root needed) to
  `~/.local/opt/llvm-21.1.8`, `make`'s LLVM_BIN there by default. Commits
  reach it by git bundle (as for training, `/data/serenity/serenity`'s
  remote), the work branch as a worktree, `/data/serenity/cuda`. It builds
  the core, the measuring program, the CUDA backend as far as its phases
  have come, the headless renderer on it, and `tests/cuda`, which run on
  its GPU; the pins there add CUDA's `nvcc` and NVIDIA's driver.
- **Python is outside the C++ gates.** `scripts/` and `training/` are
  reviewed for correctness, not against the C++ corpora; `training/` keeps
  its own pinned environment (`training/README.md`), and nothing the
  renderer builds or runs depends on it.
- **A milestone closes with its write-up, `main` and a tag.** The write-up
  goes in `docs/writeups/`, then `main` is fast-forwarded to the work
  branch, tagged `milestone-N`, and pushed with the tag
  (`git push origin main milestone-N`); `main` only ever fast-forwards.
  The tags (`milestone-1` to `milestone-5` so far) and source control are
  the archive of what each milestone built: what a later decision
  supersedes is deleted, not kept (MEMORY.md, "Decided by the user
  2026-10-17"). Milestone 3 added two steps
  before its write-up: a guidelines and performance pass over its own
  code, both corpora rule by rule, every graph shown unchanged to the bit
  (`../research/2026-10-15-milestone-3-guidelines-pass.md`), and a
  documentation pass bringing the architecture, process and front-page
  documents up to what was built.

## Formatting and static checks

Decided by the user 2026-10-19. The tools are LLVM 21.1.8's (Homebrew's,
`brew install llvm`, on the Mac; LLVM's own release on the RTX 3090 Ti's
machine, above), pinned in `cmake/toolchain.json` ("$lint") since the
formatter's output depends on its version; the build needs none of them,
and only the targets below check the pin (`tools/check_toolchain.py
--lint`).

- **Google style, unchanged.** `.clang-format` is `BasedOnStyle: Google`
  and nothing else; Objective-C++ (`display_link.mm`) takes Google's
  Objective-C column limit, 100. No style is tuned to match older code.
- **clang-tidy:** `google-*`, `cppcoreguidelines-*`, `bugprone-*`,
  `performance-*` and `clang-analyzer-*`, every finding an error
  (`.clang-tidy`), but `bugprone-easily-swappable-parameters`, off as
  clang-tidy's own configuration has it: it flags most functions of two
  parameters or more, and its fixes are mostly ceremony. Two of its fixes
  stay where a swap is a real risk: `frame::HeldSpan` (a held span's start
  and frame, two frame indices; the history test's helper beside it), and
  the ReSTIR PT test's merge settings (five floats). Nothing else is
  switched off.
- **The deviations,** each suppressed where it applies, with its reason:

  | Where | Check | Why |
  |---|---|---|
  | Headers the Metal compiler also compiles (`core/contracts/`, the lights', materials', media's, shapes' and textures' records, `bindings.h`, `scene_block.h`, the GPU tests' probes) | `avoid-magic-numbers` on the byte sizes asserted; `avoid-c-arrays`; `performance-enum-size` on 32-bit kinds | The layout the shaders read: MSL has no `std::array` in `constant` layouts, and the sizes and underlying types are that layout (AGENTS.md, ES.27, Enum.7/8) |
  | `metal/passes/bindings.h` | `use-enum-class` | `[[buffer(n)]]`, `[[texture(n)]]` and Metal's setters take an integer index (AGENTS.md, Enum.3); the enumerations are a byte each |
  | `metal/device/frame_array.h`, `view()` | `pro-type-reinterpret-cast` | Metal's buffer memory is untyped (`contents()`), viewed as the Ts copied into it |
  | `.clang-tidy`, `clang-analyzer-core.CallAndMessage:CXXThisMethodCall` off | the analyzer's null-`this` call | metal-cpp's `NS::SharedPtr` releases through a null pointer by design, Objective-C's message to nil (as `cmake/ubsan_ignorelist.txt` lets it pass the sanitizer); reported inside metal-cpp, where no line can say so |
  | `tests/.clang-tidy` | `avoid-magic-numbers` | In a test the literal is the expectation; ES.45 is about constants in program logic |
  | `frames_in_flight`, `max_texture_side`, `max_accumulated_frames` | `google-objc-global-variable-declaration` | C++ constants, reached by Objective-C's naming rule only because `display_link.mm` (Core Animation's `CAMetalDisplayLink`, which metal-cpp lacks) includes them |
  | `tests/gpu/allocation_test.cpp` | `no-malloc`, `owning-memory`, `avoid-non-const-global-variables` | It replaces operator new and delete: under them is only the C heap, and its counters are the only state operator new can reach |
  | `output/pfm.cpp`, the C stream's deleter | `owning-memory` | The file is opened by exclusive create (`std::fopen`'s `"wx"`), so an existing file, a reference above all, is never replaced; C++20's file streams cannot (`std::ios::noreplace` is C++23). The check then asks for GSL's `gsl::owner`, which the project does not use; the `unique_ptr` is the owner |

- **`make format`** formats every C++, Objective-C++ and Metal source in
  place; **`make format-check`** fails on any that is not; **`make tidy`**
  builds `native-debug` (its generated headers are read) and runs clang-tidy
  over its compile database, `src/` and `tests/`. `make check` runs
  `format-check` and `tidy` after its structural checks.
- **VS Code:** the clangd extension (`.vscode/extensions.json`), run as
  `.vscode/settings.json` says: LLVM 21's clangd with clang-tidy, formatting
  C, C++, Objective-C++ and Metal on save, Microsoft's IntelliSense off.
  `.clangd` reads `build/native-debug`'s compile database (configure it
  first: `make test`). clangd runs the checks above but the static
  analyzer's, which `make tidy` runs. Metal sources open as C++ for
  navigation and report nothing: clangd cannot compile the Metal Shading
  Language, and the build's Metal compiler is their check.

Do not weaken inherited governance without a decision; MEMORY.md records the
ones taken.
