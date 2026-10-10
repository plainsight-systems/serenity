# Sanitizers on the development machine

*2026-10-10. Apple M3 Max, macOS 26.6.2 (25G83), Xcode 26.2, Apple clang
17.0.0 (clang-1700.6.3.2). Debug builds.*

## Question

The guideline sweep asked for a build under AddressSanitizer and
UndefinedBehaviorSanitizer, with the tests run under it (P.12). Which of
them run here, and what do they find?

## AddressSanitizer: hangs before main

A program of five lines (an array read, a `printf`), built with
`-fsanitize=address`, never reaches `main`: killed by an alarm after 20 s,
every time. `sample` shows where: ASan's initializer maps its shadow
memory, asks whether the range is free (`MemoryRangeIsAvailable`), walks
the dyld shared cache for that (`get_dyld_hdr`,
`dyld_shared_cache_iterate_text_swift`), and the walk copies a block, which
calls `malloc`, which ASan's zone sends back into ASan's initializer, which
waits on the spin lock its first entry holds:

    __asan::AsanInitInternal -> InitializeShadowMemory
      -> MemoryRangeIsAvailable -> get_dyld_hdr
      -> dyld_shared_cache_iterate_text_swift -> _Block_copy -> malloc
      -> __asan::AsanInitFromRtl -> StaticSpinMutex::LockSlow (spins)

The same five lines hang built by Homebrew's clang 21.1.8 with its own
runtime, and run outside the agent's sandbox as inside it: the hang is the
runtime's on this macOS, not the pinned toolchain's alone, nor the build's.
The core's tests built this way hung the same way, for twenty minutes,
before any test ran.

So the build has no AddressSanitizer preset: one that cannot start would
claim a check that never happens. Revisit with the next Xcode (the pin
moves in a commit of its own, `cmake/toolchain.json`).

## UndefinedBehaviorSanitizer: one third-party idiom

`native-sanitize` (CMakePresets.json) builds everything, dependencies
included, with `-fsanitize=undefined -fno-sanitize-recover=all`, so the
first report stops the test that made it.

- The core's tests (113 cases): no report.
- The GPU tests (88 cases): one report, in the first test to make a
  `Submission`: `member call on null pointer of type
  'NS::Referencing<MTL4::CommandAllocator>'`, in metal-cpp's
  `NS::Referencing::release()`. Assigning to an `NS::SharedPtr` releases
  what it held, and an empty one holds null: metal-cpp sends `release` to
  nil, which Objective-C makes a no-op. metal-cpp marks the assignment
  operators that do so `no_sanitize("undefined")`, but not `release()` and
  `retain()`, which they call. Every assignment into an empty `SharedPtr`
  in `src/metal/` does it.

`cmake/ubsan_ignorelist.txt` exempts those two members, by mangled name,
and nothing else. A call this project makes on a null Metal object is
still reported: a program that assigns into an empty `SharedPtr` and then
calls `newCommandQueue()` on a null `MTL::Device*` runs past the first and
stops at the second (`member call on null pointer of type 'MTL::Device'`).
With it, `ctest --preset native-sanitize` passes: the core's tests, the
boundary and toolchain checks, and the GPU tests.
