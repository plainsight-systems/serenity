# The reference and its error: the guideline pass

*2026-10-10. A rule-by-rule pass over the implementation of the reference
design (branch `reference`, on 0601b2d): contract 13,
`src/core/film/linear_image.cpp`; `src/core/output/pfm.cpp`,
`image_format.cpp`; `src/core/measurement/reference.cpp`, `error.cpp`;
`src/headless/options.cpp`, `main.cpp` (--format, the 2^32 bound);
`src/measure/options.cpp`, `main.cpp`; the build and Makefile targets; and
the tests: `tests/linear_image_test.cpp`, `pfm_test.cpp`,
`reference_test.cpp`, `error_test.cpp`, `measure_test.cpp`, the changes to
`options_test.cpp` and `test_headless.sh`, `tests/gpu/headless_pfm_test.cpp`,
and `tests/support/files.h`, `run_program.h`, `programs.h.in`. The design is
in those headers and in [../2026-10-10-reference.md](../2026-10-10-reference.md).*

Both corpora answered for the whole pass:

- the C++ Core Guidelines, over the `cpp-guidelines` MCP server
  (`list_category` for every category below, `get_guideline` and
  `search_guidelines` for the rules cited);
- the C++ performance guidelines, over HTTP at `localhost:7015/mcp` (its MCP
  server did not connect to the session; the HTTP endpoint answered every
  call, `list_category` for all thirteen categories).

Every category of both was walked. For each: the rules that bear on the
diff, each finding with its rule and verdict, and the rules checked and
found clean. A verdict is **fixed** (in this change), **convention** (a
project-wide departure recorded in AGENTS.md, "Guideline deviations"), or
**rejected** with the rule-based reason.

## Summary

| Verdict | Count |
|---|---|
| Fixed | 10 |
| Convention (AGENTS.md) | 2 |
| Rejected, with reason | 13 |

The fixed findings that changed a reviewed header: ES.3 (contract 13 gains
`checked_values()` and `value_position()`) and I.24 (`error_against()` takes
a `Judged`, its two images by name). Both are listed, with the
implementation's other header changes, in the design note's
[Implementation](../2026-10-10-reference.md#implementation).

## C++ Core Guidelines

### P (Philosophy)

- Clean. P.1 (a reference's state is a class, `ReferenceBuilder`; a command
  a `std::variant`), P.4 and P.5 (`static_assert`s on little-endian and
  32-bit IEEE floats in pfm.cpp, so a byte order this reader does not swap
  fails to compile), P.6 and P.7 (every file, extent and value checked
  before use, and the command line before any file is read), P.8 (streams
  and vectors only), P.9 (no copy of an image to write or read it).
- **Rejected:** P.9, `read_pfm()` zero-fills the image before the read
  overwrites it (`std::vector` value-initializes). One pass of stores over
  an image read once per batch; avoiding it needs a default-initializing
  allocator (MEM.8) for a path that is not performance-sensitive (Per.3).

### I (Interfaces)

- **Fixed:** I.24, `error_against(image, reference)` took two
  `LinearImage`s side by side, and the measure is asymmetric (the relative
  MSE divides by the reference's values, and only the reference's are
  refused below 0): swapped arguments gave other numbers silently. It takes
  a `Judged {image, reference}`, named at every call, as the codebase does
  with `RunFrame` and `HeadlessRun`.
- I.5 and I.10: preconditions stated in each header and checked by throws
  (`std::invalid_argument`, `std::logic_error`, `PfmError`, `OptionsError`),
  before any state changes. I.30: the two `reinterpret_cast`s are kept in
  two named helpers in pfm.cpp (see ES.48). I.4: `ImageFormat` is an
  `enum class`, not a string past parsing.
- **Rejected:** I.6 and I.8 (`Expects`, `Ensures`): this repository has no
  GSL; preconditions are checked by explicit throws, as everywhere in
  `src/` (I.5, E.2).
- Checked and clean: I.1, I.2, I.3, I.11 to I.13, I.22, I.23, I.25 to I.27.

### F (Functions)

- **Fixed:** F.3, `read_pfm()` read the header, checked the length, read,
  checked again and turned the rows in some sixty lines; the header is now
  `read_header()`, returning what it says (F.21: a struct, `Header`).
- F.16: `Judged`, two references, is passed by value; images by
  `const&`; paths by `const&`. F.20: every result returned. F.52: the
  headless renderer's `write_frame` lambda captures by reference, used in
  place.
- **Rejected:** F.6, `image_format_named()` and `extension()` are not
  `noexcept`: neither must not throw (the second throws for a value no kind
  names), and the reviewed header declares them so.
- Checked and clean: F.1, F.2, F.4, F.8, F.9, F.15, F.17 to F.19, F.22 to
  F.27, F.42 to F.56, F.60.

### C (Classes)

- C.2: `ReferenceBuilder` is a class (its arrays sized by its extent once a
  batch is in); `LinearImage`, `ImageError`, `Judged`, `Header`,
  `MakeReference`, `MeasureError` are structs of values. C.4: the floors are
  members, needing the state. C.20: no special members. C.181 and C.182:
  the command is a `std::variant`, visited with a generic lambda (C.170).
  C.90: memcpy only in tests, on floats.
- **Rejected:** C.12, `Judged` holds two references, so it cannot be
  assigned. It is a parameter, built for one call and never stored; a
  pointer would admit null (F.60, I.12), and a value would copy two images
  of some 25 MB each.
- Checked and clean: C.1, C.3, C.5, C.8, C.9, C.10 to C.13, C.21, C.30 to
  C.67 (none defined), C.80 to C.90, C.120 to C.183.

### Enum (Enumerations)

- Clean. Enum.3 (`enum class ImageFormat`), Enum.7 and Enum.8 (no
  underlying type or values given), ES.79's companion: both switches over it
  have no `default`, so a new kind without a case fails to compile.

### R (Resource management)

- Clean. R.1 (`std::ofstream`, `std::ifstream`, `std::vector`, and the
  tests' `ScratchDirectory`, which removes its directory however a test
  ends), R.5 (no heap object), R.10, R.11 (no `new`, `malloc`).

### ES (Expressions and statements)

- **Fixed:** ES.3, contract 13's invariant was checked in four places
  (`write_pfm()`, `ReferenceBuilder::add()`, `error_against()` twice) and
  "pixel (x, y) channel c" written twice. Both are now the contract's,
  `checked_values()` and `value_position()`, defined by Film, its owner.
- **Fixed:** ES.45, the accumulated image's 4 floats a pixel in the headless
  renderer, and the PFM header's 3 newlines, are named constants.
- **Fixed:** ES.105, `value_position()` divides by the width; a width of 0
  is refused by `std::invalid_argument` rather than divided by.
- **Fixed:** ES.79, a comment in image_format.cpp cited Enum.2 for "no
  default"; the rule is ES.79.
- ES.103: every size is computed from sides already bounded by
  `max_image_side`, and the headless bound's sum and product are each
  bounded before the next is taken; the tests feed 2^64 - 1 to both. ES.46:
  every narrowing is a named cast with its reason (`stream_size`, the side
  read, the batch count as a double). ES.20, ES.25, ES.71 (index loops only
  where one index addresses two arrays), ES.86 (the argument cursor, as the
  headless parser's).
- **Rejected:** ES.48 and ES.49, the type profile's ban on
  `reinterpret_cast`: pfm.cpp has two, `const float*` and `float*` to
  `char*`, for the streams' `read` and `write`. `char` may alias any object,
  the floats exist before they are read into (LIFE.4), and the alternative,
  a byte buffer copied to and from, doubles a read's memory past the
  contract's 3 GiB bound. Encapsulated in two helpers (I.30).
- **Rejected:** ES.102 and ES.107, unsigned subscripts: indices are
  `std::size_t`, the containers' own `size_type`, as in the rest of `src/`;
  signed indices would mix signedness at every subscript (ES.100), and
  every index is bounded by a checked count first (SL.con.3).
- **Rejected:** ES.48's note on `(void)`: the discarded values
  (`checked_values()`'s count) are not `[[nodiscard]]`, and `(void)` is the
  codebase's idiom for a value deliberately unused.
- Checked and clean: ES.1, ES.2, ES.5 to ES.12, ES.21 to ES.24, ES.26 to
  ES.34, ES.40 to ES.44, ES.47, ES.50, ES.55, ES.56, ES.60 to ES.65, ES.70,
  ES.72 to ES.78, ES.84, ES.85, ES.87, ES.100, ES.101, ES.104, ES.106.

### Per (Performance)

- Per.3: none of this is on a frame's path; the costs are stated in the
  headers (one pass of some 20 operations a value a batch, tens of
  milliseconds at 1920 x 1080). Per.14: the reference's state is allocated
  once (MEM.9), the headless renderer's readback buffer once a run, and a
  batch is never held beside another. Per.19: every loop walks its arrays
  in order.
- Checked and clean: Per.1, Per.2, Per.4 to Per.7, Per.10 to Per.13,
  Per.15 to Per.18, Per.30.

### CP (Concurrency)

- Clean, not applicable: nothing here runs on another thread. CP.1:
  `ReferenceBuilder` is a value with no shared state; two builders on two
  threads share nothing.

### E (Error handling)

- **Fixed:** E.14, serenity-measure refused an existing --out with
  `std::runtime_error`; it throws `measure::OptionsError`, as the headless
  renderer's `prepare_output()` refuses a directory that holds anything.
- E.2 and E.3: every refusal is an exception naming the file, the argument
  or the pixel; a failed run prints only its message (measure/main.cpp
  builds its output whole first). E.4: `ReferenceBuilder::add()` checks the
  whole batch, then allocates aside and moves in, then folds with nothing
  that can throw: a refused batch leaves the state as it was (the tests
  check the bits). E.6: streams close themselves. E.28: numbers are read
  by `std::from_chars`, no `errno`.
- **Rejected:** a residual window, not a rule broken: C++20 has no
  exclusive-create open (`std::ios::noreplace` is C++23), so a file made
  between serenity-measure's check and its write would be truncated. The
  check is made before the batches are read and again just before the write,
  which narrows the window to the write itself; a POSIX `O_EXCL` open would
  bring a platform call into a core-only program (E.2's purpose is met for
  every case but a race with another writer).
- Checked and clean: E.1, E.5, E.7, E.8, E.12, E.13, E.15 to E.19, E.25 to
  E.27, E.30, E.31.

### Con (Constants)

- Clean. Con.1 and Con.4 (`const` locals throughout), Con.2 (the builder's
  readers are `const`), Con.3 (images by `const&`), Con.5 (`constexpr`
  bounds and channel counts).

### T (Templates)

- Clean. T.141: unnamed lambdas used in one place (the visit, the tests'
  refusal helpers). No template is added; the tests reuse `error_of<E>`.

### SF (Source files)

- **Fixed:** SF.6, measure/main.cpp had `using namespace serenity;` at file
  scope; its helpers now live in `serenity::measure`, and `main()` uses a
  namespace alias.
- **Fixed:** SF.10, names reached through other headers' includes:
  `<string>` in pfm.cpp, `<filesystem>`, `<span>` and `<vector>` in
  measure/options.cpp are included where used.
- **Convention:** SF.8, `#pragma once` in the new test headers.
- **Convention:** SF.12, quoted includes through `-I src` (and the tests'
  generated `"programs.h"`, as `"test_paths.h"`).
- SF.5: each .cpp includes its header first. SF.22: helpers in unnamed
  namespaces. SF.11: every new or changed header compiles alone (each
  checked with `clang++ -std=c++20 -fsyntax-only` and the source warnings).
- Checked and clean: SF.1 to SF.4, SF.7, SF.9, SF.13, SF.20, SF.21.

### SL (Standard library)

- SL.io.2: every PFM is untrusted; each header line is bounded before it is
  kept, the file's length is checked against the header before anything is
  allocated, and the read is checked again for a file that changed. SL.io.3:
  iostreams. SL.io.50: `'\n'` and an explicit flush, no `endl`. SL.con.3:
  every index follows `checked_values()`. SL.str.2: `std::string_view` for
  names passed in.
- **Rejected:** SL.io.1, the header is read a character at a time: that is
  where it has to be, for `std::getline` would read a binary file's first
  "line" whole, whatever its length, before any bound could apply.
- **Rejected:** SL.io.10, `sync_with_stdio(false)` is not called:
  serenity-measure prints a few lines once (Per.3).
- Checked and clean: SL.1 to SL.4, SL.C.1, SL.con.1, SL.con.2, SL.con.4,
  SL.str.1, SL.str.3 to SL.str.12.

### NL (Naming and layout)

- **Fixed:** layout, lines past the codebase's 120 columns in pfm.cpp and
  three tests, wrapped; an include out of order in headless/main.cpp,
  sorted (NL.4's consistency).
- NL.10 (underscore_style), NL.8 (the codebase's naming), NL.2 (comments
  state intent and cite their rules), NL.16, NL.17, NL.18, NL.20, NL.21,
  NL.26, NL.27.
- Checked and clean: NL.1, NL.3, NL.5, NL.7, NL.9, NL.11, NL.15, NL.19,
  NL.25.

### A, CPL, NR, GSL

- A.2 and A.4: serenity-measure's command line is a library of its own, as
  the other two programs'; it depends on the core alone, and the boundary
  check now enforces that (rule 2b). CPL: C++ throughout; the tests' one
  use of `std::system` and the `WIFEXITED`/`WEXITSTATUS` macros is kept in
  `tests/support/run_program.h` (P.11), its arguments single-quoted and
  checked for quotes. NR.5: no two-phase initialization (the builder is
  usable empty, its refusals explicit). GSL: not a dependency here (I.6
  above).

## C++ performance guidelines

### memory (MEM)

- MEM.9: the reference's state, 48 bytes a pixel, allocated once at the
  first batch; the headless renderer's readback buffer once a run.
- **Rejected:** MEM.1 to MEM.8, MEM.10, MEM.11, custom allocators and
  arenas: a few allocations a run on a path that is not
  performance-sensitive (Per.3), through `std::vector` as everywhere in the
  core.

### copy-move (COPY)

- Clean. COPY.7: images are read by `const&` (the range-for over paths
  too), and `builder.add(read_pfm(batch))` binds the temporary. COPY.8:
  every image is returned by value, named or not. COPY.6: memcpy only on
  floats, in tests. COPY.1: no `std::move` on return.

### cache-layout (CACHE)

- CACHE.3: contiguous arrays only.
- **Rejected:** CACHE.4, the Welford state is two arrays, mean and M2, both
  read and written in the same loop; interleaving them would make one
  stream of two. Tens of milliseconds a batch beside seconds of rendering
  (Per.3).
- Not applicable: CACHE.1, CACHE.2, CACHE.5 to CACHE.8.

### lifetime (LIFE)

- LIFE.4: `read_pfm()` writes the bytes of floats that already exist, made
  by `make_linear_image()`, through `char*`: no object's lifetime is begun
  from raw storage. Not applicable: LIFE.1 to LIFE.3, LIFE.5 to LIFE.8.

### concurrency (CONC)

- Not applicable: single-threaded.

### codegen (GEN)

- Not applicable: no hot path. GEN.7's cold paths are refusals that end a
  run.

### gpu (GPU)

- GPU.1: --format pfm reads the accumulated image back once a written
  frame, 16 bytes a pixel, as the design budgets it
  (2026-10-10-reference.md); a frame not written is not read back.
- Not applicable: GPU.2 to GPU.10 (no kernel or dispatch added).

### gpu-dsa (GDSA)

- GDSA.2: each floating-point sum states its determinism, run to run, in
  its header and at the loop: the reference's fold in the batches' order,
  the error's and the floors' in pixel order. The tests check the bits.
- GDSA.3: every random number is keyed by an index, a sample's frame index
  on the GPU and `draw(seed, batch, value)` in the tests; the headless bound
  keeps those indices distinct.
- Not applicable: GDSA.1, GDSA.4 to GDSA.21.

### cpu-dsa (CDSA)

- CDSA.23: the reference is a streaming state, (n, mean, M2) in double,
  with Welford's update and a fixed combine order.
- **Rejected:** CDSA.23's merge (Chan, Golub and LeVeque): there is one
  writer and the batches arrive one after another, so no two states are
  ever combined; a merge would be code with no caller.
- Not applicable: CDSA.1 to CDSA.22, CDSA.24 to CDSA.33.

### telemetry (TLM), simd (SIMD), embedded (EMB), wasm (WASM)

- Not applicable: no instrumentation, no vector code, no embedded or
  WebAssembly target.

## The review of 5f4c7de

Codex reviewed 5f4c7de, both focuses (`.cache/reviews/5f4c7de-*.json` in
the main checkout). Each finding, the rule it falls under, and its
verdict; both corpora consulted again for the code each fix writes. The
design note's [Review of 5f4c7de](../2026-10-10-reference.md#the-review-of-5f4c7de)
keeps the account.

| # | Finding | Rule | Verdict |
|---|---|---|---|
| 1 | parse() bounded the last sample index at --samples, refusing a graph that accumulates nothing, which renders one sample a frame (`--first 4194304 --samples 1024` of the test pattern); the shell test pinned the false refusal | I.5 (a precondition checked against what is actually run) | **Fixed.** parse() bounds the last frame index; `check_samples()` bounds the plan's samples, after the graph is read and before anything renders or is made. The wrap test now uses the path graph; the test pattern's run is accepted and written |
| 2 | "A reference is never replaced" was a check, then a truncating open: two runs could both pass it | E.2 (the contract as stated must hold), CP.2 (two writers, one file) | **Fixed.** `write_pfm()` opens by exclusive create (`std::fopen(path, "wbx")`), the check itself; a partial file of a failed write is removed. serenity-measure's own check is gone |
| 3 | The headless renderer enumerated Output's kinds: their semantics in its header, names in its parser, a branch per writer in its main | A.1, ES.3 (one place names a kind) | **Fixed.** image_format.h lists each kind's names, extension and source in one table; `write_image()` writes any kind; the headless renderer reads back by source and calls it |
| 4a | The Makefile said "most of an hour" for the default reference | Per.6 (no claim without its count) | **Fixed.** "Some ten minutes at 1920 x 1080, scaling with pixels and samples", pointing at the note |
| 4b | "One readback ... copied once": it is three copies, 16, 16 and 12 bytes a pixel | GPU.1, GDSA.6 (count the passes over memory) | **Fixed,** the note and the headless renderer's comment give all three, 33, 33 and 25 MB at 1920 x 1080. No optimization added: a fraction of a second a reference (Per.3) |
| 4c | error.h said one pass; it is two | Per.6, GDSA.6 | **Fixed,** error.h says two and why; reference.h states its two passes a batch the same way. Fusing them was offered and is **rejected**: the header promises every refusal before any sum, and the second pass costs 50 MB of reads beside seconds of rendering (Per.3) |
| 5a | Frozen time rebuilds the same acceleration structure every sample | GDSA.17 (plan once per key) | **Rejected,** as at the design's review (2026-10-10-reference.md, "Review"): some 2% of a reference, and caching the scene by time touches every run's frame loop. Its measured share comes with the first references |
| 5b | A PFM run tone maps every sample into an image it never reads | GPU.6, GDSA.6 | **Rejected,** as at the design's review (same section): some 2%, and a graph ending after the path pass changes the rule that a graph computing light ends in a presenting pass. Measured with the first references |

The code each fix writes, rule by rule:

- **R.1, R.20, E.6:** the C stream is a `std::unique_ptr<std::FILE,
  CloseFile>`, closed however its scope ends; the writer closes it
  explicitly to see the close's result (a buffered write can fail there).
  The tests' `FileSizeLimit` restores the file-size limit and the signal it
  ignores the same way.
- **SL.io.3, rejected:** the writer uses a C stream, not an iostream: C++20's
  file streams have no exclusive create (`std::ios::noreplace` is C++23).
  Kept to pfm.cpp, stated in pfm.h (I.30). It also takes the floats as they
  are, so the writer's `reinterpret_cast` is gone; the reader's one remains
  (ES.48, rejected as before).
- **E.28:** why an exclusive open failed is asked of the file system
  (`symlink_status`), not read from `errno`.
- **ES.3, ES.27, SL.con.1:** the kinds are one `std::array` of rows; every
  lookup reads it. **ES.79:** `write_image()`'s and the headless
  renderer's switches have no `default`, so a new kind or source without a
  case fails to compile; a value no kind names throws `std::logic_error`
  (P.6). **ES.78:** no fallthrough: the refusing cases call a
  `[[noreturn]]` function.
- **Enum.3:** `ImageSource` is an `enum class`. **I.24:** `write_image(format,
  path, image)` and `check_samples(options, samples_per_frame)` take
  parameters of distinct types. **I.5:** `check_samples()` refuses a
  samples count no plan makes, by `std::invalid_argument`.
- **ES.103:** the bound's sum and product, now one function
  (`past_last_sample`) for both checks, each step bounded before the next.
- **CP.2 in the tests:** the file-size limit and the ignored SIGXFSZ are
  process state, set and restored inside one test case; doctest runs one
  case at a time.

Shown to fail, each by breaking the code it guards, then restoring it:
the bound checked at --samples, or not at all, in the headless renderer
(test_headless.sh); `check_samples()` checking nothing, and parse()
dropping the frame bound (options_test); a non-accumulating graph's
accumulated kind not refused (test_headless.sh); the truncating open
instead of the exclusive one (pfm_test and measure_test); the partial file
left, and both the write's and the close's failure ignored (pfm_test); an
accumulated image written as a PNG, pfm's source given as displayed (the
Output tests, and the headless run), the names list mis-joined, a PFM of
zeros written by `write_image()` (image_format_test); and the headless
renderer writing zeros for the accumulated image (the GPU test). Ignoring
only the close's failure is not caught alone: under the file-size limit the
write itself fails first, so the two checks are shown together.
