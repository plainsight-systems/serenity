# Milestone 3 against the two guideline corpora, rule by rule

*2026-10-15. Milestone 3's code on branch `denoiser`, from 139ed5d: every
C++ and Metal file `git diff main..139ed5d` adds or changes under `src/`
and `tests/` (the features pass, the two-resolution frame, the denoise
pass and its kernels, the network loader and JSON reader, contracts 17 and
18, the headless and renderer changes, their tests), each read in context;
and a lighter review of its Python (`training/`, `scripts/
make_denoiser_dataset.py`, the change to `scripts/make_marbles.py`).
Milestones 1 and 2's code that Milestone 3 did not touch is out of scope;
a codebase-wide pass is queued (QUEUE.md). Both corpora answered through
their MCP servers (`cpp-guidelines`, `cpp-perf-guidelines`); every category
of each was listed fresh and walked over the scope; every finding has an
ID and a verdict, and every valid one is fixed. Apple M3 Max, the
toolchain in `cmake/toolchain.json`.*

## Behavior unchanged

- **Byte for byte.** `serenity-headless`, release, before (139ed5d) and
  after, on the marbles at 480 × 270, frames 40 and 41: PNG of
  test_pattern, preview, path, path_mnee, restir, restir_ris,
  restir_spatial, restir_temporal and restir_pt; PFM of the seven that
  accumulate; restir_pt_features as PNG and PFM with `--features`; the
  features graph alone with `--features`; restir_pt_denoised (the wide
  network) as PNG and PFM with `--features`; restir_pt_denoised_small as
  PFM; and restir_pt_denoised at 1920 × 1080, frames 120 and 121, PFM.
  84 files; `diff -r` finds none differ, for the build of the fixes and
  again for the build with the kept performance change.
- **The match against PyTorch.** The denoised PFMs above are the same to
  the bit, so the denoise pass computes what it did, and its distance from
  PyTorch is the one the Phase 4 note measured (2.2 × 10^-4 on the
  fixture; on real frames, the note's table). `tests/gpu` denoise test
  green in all three presets (below). `training/denoise_check.py` on the
  real frames was not rerun: no number of any kernel changed (above).
- **The gates.** All three presets build with 0 warnings; ctest green on
  each; `make check` green.

## C++ Core Guidelines: findings and verdicts

Lines are 139ed5d's.

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| G1 | F.24, I.13, P.9 | core/denoise/network.h:89, network.cpp:172-184, 230-236 | `parse_network` took `const std::vector<std::byte>&`; `read_bytes` read each file into a `vector<char>` and copied it into a `vector<std::byte>`, and `load_network` copied layout.json's bytes again into a string | fixed: `parse_network(…, std::span<const std::byte> weights, …)`; each file read once into a string, the weights viewed as bytes where they are (`std::as_bytes`); the test's copy helper gone with it |
| G2 | SF.11 | core/denoise/network.h | named `std::byte` without `<cstddef>` | fixed (with `<span>`) |
| G3 | ES.45, NL.11 | network.cpp:33, 41-44 | `9007199254740992.0` and `4096` as literals | fixed: `largest_exact_count` (2^53), `max_width`, the message built from it |
| G4 | ES.3 | network.cpp:97-99; denoise.metal:539 | the spacings 1, 2, 4 written twice, in the loader's check and the shader's table | fixed: `contracts::spacing_pixels` in contract 18, read by both |
| G5 | ES.3 | core/camera/thin_lens.cpp:98-100, 118-121 | `view_axes` repeated `shader_form`'s forward and right | fixed: one `basis()`; the same operations in the same order, so `shader_form`'s bits are unchanged (byte identity above) |
| G6 | ES.3, SF.3 | metal/frame/feature_images.cpp:19-22, passes/denoise/denoise.cpp:20-24, tests/gpu/support/denoise_fixture.h:83 | `using Half = _Float16` declared three times, one comment pointing at another | fixed: once, in `metal/device/support.h`, which all three include |
| G7 | ES.45 | denoise.cpp:49-51, 128-132, 161-163, 171-173, 195-196; denoise.metal:297-301, 474 | the convolution's and output kernel's threadgroup shape (32 pixels, 4 rows, 128 threads, 128 head columns) written as literals on the host and again in the shader, each side free to drift from the other | fixed: `simd_width`, `conv_rows`, `conv_pixels`, `denoise_threads`, `head_columns` in contract 18, read by both sides, with `static_assert`s for what the kernels assume (a row's pixels are a simdgroup's lanes; the output kernel's simdgroups are the subpixels; the head's columns are a simdgroup's width each); the median's and pool's host-only sizes named in denoise.cpp |
| G8 | ES.45 | metal/frame/denoise_images.cpp:91-164; denoise.cpp:175-179 | the U-Net's four levels and the dispatch order's indices (4, 5, 6 + i, 10) as literals | fixed: `DenoiseImages::levels` (`contracts::unet_levels`), `first_bottleneck`, `first_decoder`, `last_convolution`, `convolutions` derived from them |
| G9 | ES.45 | denoise.metal:135-186, 249-268, 313-409, 497-522 | the channel block `8u`, the median's bit split (`20`, `10`, `1023`, `1024`) and the input channels' places (`12 + q`, `16 + q`, `20 + q`, `24 + q`) as literals | fixed: `block` (`contracts::channel_block`), `top_shift`, `level_bits`, `level_bins`, `level_mask`; `contracts::input_radiance` … `input_validity`, asserted to fill the 28 channels |
| G10 | ES.30, ES.31 | denoise.metal:413-421 | `SERENITY_CONV3X3`, a macro pasting the kernel's four instantiations | fixed: written out, as the median's six are |
| G11 | ES.48 | denoise.metal:343, 348, 361, 444, 448 | device buffers of halves read and written through `reinterpret_cast<device half4*>` | fixed: the buffers that are only ever read or written a half4 at a time are typed `half4` (conv3x3's input, skip and weights; the max pool's both; with P2, the input kernel's and conv3x3's outputs), indexed in half4s. Kept, stated in the kernel: one reinterpretation of each threadgroup tile, written as half4s and read by `simdgroup_load`, which takes halves |
| G12 | ES.3 | denoise.metal:255-257 | a `Float3` widened by hand, `float3(k.right.x, k.right.y, k.right.z)`, where `to_float3` (metal/device/layout.metal.h) is the codebase's one way | fixed |
| G13 | ES.3 | denoise.metal:545, 552 | a tap's position computed by the same expression in both softmax loops | fixed: `tap_at()` |
| G14 | ES.20, ES.22, F.3 | passes/features/features.metal:200-219 | `Found found;` declared without a value and assigned in three branches of an 85-line kernel | fixed: `replaced()`, which returns from each branch; `const Found found = replaced(…)` |
| G15 | ES.3 | metal/frame/renderer.cpp:653-660, 715-721 | the features pass's constants built and written twice, at the light size and at the output size | fixed: `write_feature_constants()` |
| G16 | NL.2, NL.3 | renderer.cpp:781-782; passes/denoise/denoise.h:38 | the comment said the denoise pass's first dispatch is after a barrier of its own; it is not (its first histogram follows the renderer's barrier before the pass, beside the full-size albedo's dispatch, which only its last dispatch reads) | fixed: both comments say what the barriers are |
| G17 | F.3 | passes/denoise/denoise.cpp:107-197; denoise_images.cpp:83-183 | `DenoisePass::record` (90 lines, four steps) and `DenoiseImages::layout` (100 lines: allocation, then dispatch planning) | fixed: `record_median`, `record_input`, `record_unet`, `record_output` (the header's four steps); `allocate()` and `plan()` |
| G18 | SL.str.2 | headless/main.cpp:159 | the feature names as `const char*` | fixed: `std::string_view` |
| G19 | SF.10 | tests/denoise_core_test.cpp; tests/gpu/denoise_test.cpp | `std::array`, `std::move`, `std::uint32_t` and `std::span` used without their headers; `std::max` without `<algorithm>` | fixed; the one refusal checked by `find() != npos` now `tests::contains` as its neighbors |
| G20 | ES.45 | tests/gpu/features_test.cpp:52 | π as a 21-digit literal | fixed: `std::numbers::pi` |
| G21 | ES.45 | tests/gpu/support/denoise_fixture.h:186-187 | the argument table's sizes as 12 and 8 | fixed: from `bindings::denoise` |

Judged not valid, or a recorded convention:

| # | ID | Where | Why |
|---|---|---|---|
| G22 | I.23 | network.cpp, `tensor()` | six parameters, the limit Phase A's pass held to (`shade`'s six); each is a different kind of thing, none adjacent of one type (I.24) |
| G23 | Con.2 | feature_images.h, `read() const` | const for the images it reads; the submission it records the copy into is a dependency held by reference, not the object's state |
| G24 | C.2, ES.3 | renderer.h, `denoises_` beside `schedule_` | both set once at construction and never changed; `denoises_` is read in the member initializer of the argument table, before the body |
| G25 | F.1 | json.cpp, `Reader::peek()` | skips whitespace before looking; the recursive-descent reader's idiom, said in its comment; no caller expects otherwise |
| G26 | ES.27, SL.con.1 | contracts/denoise.h, `spacing_pixels[]`, the layouts' `padding[]` | convention (AGENTS.md): C arrays in layouts and constants shared with Metal |
| G27 | ES.23, ES.49, ES.64 | the shaders' `float3(x)`, `uint(x)`, `as_type` | convention (AGENTS.md) |
| G28 | ES.100, ES.102, ES.107 | the shaders' `uint` indices | convention (AGENTS.md) |
| G29 | SF.8, SF.12 | every header | convention (AGENTS.md) |
| G30 | Enum.7, Enum.8 | bindings.h's `features` and `denoise` enums | convention (AGENTS.md): their values are the shaders' binding slots |

### Every category, rule by rule

Notation as Phase A's pass: **holds** (applies, and the code keeps it), a
finding number, **convention**, or **n/a** with why.

**In** (listed fresh: In, P, I, F, C, Enum, R, ES, Per, CP, E, Con, T,
CPL, SF, SL, A, NR, NL, GSL, Tools, FAQ, Discussion) — In.0 n/a.

**P** — P.1 holds (contract 18 now names every shared number, G7-G9).
P.2 holds but `_Float16`, a stated compiler extension (G6). P.3 holds.
P.4 holds (layouts asserted both sides). P.5 holds (new pass kinds fail
every switch; the kernels' assumptions are `static_assert`s, G7). P.6
holds (an export checked against contract 18 at load). P.7 holds (graph,
schedule and network refused before the first frame). P.8 holds. P.9 G1.
P.10, P.11 hold. P.12 holds (ctest under Metal's validation, UBSan). P.13
n/a.

**I** — I.1, I.2 hold. I.3 n/a. I.4 holds (`PreviousFrame`, `ViewAxes`,
`Conv`, `Resample`). I.5 holds (preconditions in the headers; `record()`
refuses missing inputs). I.6, I.8 n/a. I.7 holds (the PyTorch match, the
history tests). I.9 n/a. I.10 holds (`NetworkError`, `JsonError`,
`MetalError`). I.11, I.12 hold. I.13 G1. I.22 holds. I.23 G22. I.24 holds
(`PreviousFrame` named for it). I.25-I.27 n/a. I.30 holds (the half4
tiles' one cast, G11).

**F** — F.1 holds but G25. F.2 holds. F.3 G14, G17. F.4 holds
(`align`, `padded`, `groups` constexpr). F.5 holds. F.6 holds (address
getters `noexcept`). F.7 holds. F.8 holds (`light_size`, `view_axes`,
`laid_out`). F.9 holds. F.10 G13, G15. F.11 holds. F.15, F.16 hold.
F.17 holds (`tensor`'s `taken`). F.18, F.19 n/a. F.20, F.21 hold
(`Read`, `Found`, `Moved`). F.22, F.23 n/a. F.24 G1. F.25 n/a. F.26,
F.27 n/a. F.42-F.49 hold. F.50-F.54 hold (lambdas local, by reference).
F.55 n/a. F.56 holds. F.60 holds.

**C** — C.1, C.2 hold but G24. C.3-C.5 hold. C.7-C.9 hold. C.10-C.13
hold. C.20, C.21 hold (`FeatureImages`, `DenoiseImages` delete copy and
move). C.22 holds. C.30-C.37 hold (RAII members, `Resident`). C.40-C.49
hold (`DenoisePass` fully built in its constructor). C.50-C.52 n/a.
C.60-C.67 n/a. C.80, C.81 hold. C.82-C.89 n/a. C.90 convention (shared
layouts written by `memcpy`, trivially copyable asserted). C.100-C.183
n/a (no containers, hierarchies, operators or unions).

**Enum** — Enum.1-Enum.6 hold. Enum.7, Enum.8 G30.

**R** — R.1 holds. R.2, R.3, R.4 hold (`Needed::into`, `FrameResources`'
raw pointers non-owning). R.5 holds. R.6 holds. R.10-R.15 hold. R.20-R.24
hold (`unique_ptr` members). R.30-R.37 hold (P1 removed the one smart
pointer returned by value for a borrow).

**ES** — ES.1, ES.2 hold. ES.3 G4, G5, G6, G12, G13, G15. ES.5-ES.12
hold. ES.20 G14. ES.21 holds. ES.22 G14. ES.23 G27. ES.24 holds. ES.25,
ES.26 hold. ES.27 G26. ES.28 holds. ES.30, ES.31 G10. ES.32-ES.34 hold.
ES.40-ES.44 hold. ES.45 G3, G7, G8, G9, G20, G21. ES.46 holds (counts
range-checked before narrowing; halves stated). ES.47 holds. ES.48 G11.
ES.49 G27. ES.50 holds. ES.55 holds. ES.56 holds. ES.60-ES.63 hold. ES.64
G27. ES.65 holds. ES.70-ES.79 hold (switches over `PassKind` without
`default`). ES.84-ES.87 hold. ES.100 G28. ES.101 holds (the median's bits
unsigned). ES.102 G28. ES.103-ES.105 hold (pixel counts in `size_t`; the
softmax's sums at least 1, the median's count tested). ES.106, ES.107
G28.

**Per** — Per.1-Per.3 hold. Per.4, Per.5 hold. Per.6 holds (every
performance claim measured, below and in the Phase 4 note). Per.7 holds.
Per.10, Per.11 hold. Per.12, Per.13 hold. Per.14, Per.15 hold (buffers
made at a size, never in a frame). Per.16-Per.19 hold. Per.30 n/a.

**CP** — n/a (no CPU threads; the GPU's ordering is the barriers', GPU.8
below).

**E** — E.1-E.8 hold (load and validation errors thrown before the first
frame). E.12-E.19 hold. E.25-E.31 n/a.

**Con** — Con.1 holds. Con.2 G23. Con.3-Con.5 hold.

**T** — T.1, T.2 hold (`conv3x3<NT>`, `widened`). T.10 n/a (shader
templates; `widened`'s `Pick` a local lambda). T.40-T.144 n/a.

**CPL** — n/a.

**SF** — SF.1, SF.2 hold. SF.3 G6. SF.4, SF.5 hold. SF.6, SF.7 hold
(`using namespace metal` only in kernel files). SF.8 G29. SF.9 holds. SF.10
G19. SF.11 G2. SF.12 G29. SF.13 holds. SF.20-SF.22 hold.

**SL** — SL.1, SL.2 hold (no new dependency; JSON read by our own 200
lines, stated in json.h). SL.3, SL.4 hold. SL.C.1 n/a. SL.con.1 G26.
SL.con.2, SL.con.3 hold (`at()` on `convs_`, `pools_`). SL.con.4 holds
(trivially copyable asserted). SL.io.1-SL.io.50 hold (files read whole,
ill-formed input refused, json.h). SL.str.1 holds. SL.str.2 G18.
SL.str.3-SL.str.12 hold. SL.str.5 holds (`std::byte` for weights).

**A** — A.1, A.2, A.4 hold (loader and contracts in the core, kernels in
the backend; `make check` holds the boundary).

**NR, NL, GSL, Tools, FAQ, Discussion** — NL.1 holds. NL.2, NL.3 G16.
NL.4-NL.27 hold (names say what; `block` for the channel block in the
shader, its comment says so). NR.1-NR.7 hold. GSL, Tools, FAQ, Discussion
n/a.

## Performance guidelines: findings and verdicts

| # | ID | Where | Finding | Verdict |
|---|---|---|---|---|
| P1 | COPY.7 | passes/denoise/denoise.cpp:100-105, 159 | `pipeline()` returned `NS::SharedPtr` by value, a retain and a release (two atomic operations) for each of the 11 convolutions a frame, for a pointer only borrowed | fixed: returns the raw, non-owning pointer; the pass owns it |
| P2 | GPU.2 | denoise.metal:271-273, 401-407 | the input kernel wrote a pixel's 32 channels as 32 scalar half stores, and conv3x3's epilogue each lane's 8 channels as 8 | fixed, measured: 8 and 2 half4 stores, the same values (byte identity above); the pass 0.03 to 0.05 ms faster, nothing slower (Timings) |

Recorded as holding (the code keeps them; nothing to fix):

| # | ID | Where | Why it holds |
|---|---|---|---|
| P3 | GPU.8 | renderer.cpp; denoise.cpp | one barrier before the pass (it reads the light and the features), each of the pass's dispatches after one on the one before it, which it reads; the full-size albedo's dispatch beside the median's, which do not read it |
| P4 | GPU.9, MEM.9 | denoise_images.h | 0.95 GB unaliased at the window's light, stated in the header with what aliasing would save (0.12 GB); made at a size, before the submission, never in a frame |
| P5 | GDSA.2, GDSA.5 | denoise.metal, the median | integer atomics, aggregated per threadgroup before the device's bins; the result exact and order-free |
| P6 | GDSA.16, GPU.5 | denoise.metal | the halo and weights staged once a threadgroup; the head's scores never in device memory (measured in the Phase 4 note) |
| P7 | GDSA.17, CDSA.32 | denoise_images.cpp; denoise.cpp | every dispatch's parameters written once a size; the weights laid out once, at construction, as the kernels read them |
| P8 | GPU.3 | denoise.metal, conv3x3 | at most 4 output blocks a simdgroup, past which accumulators spill (measured, Phase 4) |
| P9 | GPU.1 | feature_images.cpp | the features read back only for `serenity-headless --features` |

### Every category, rule by rule (listed fresh: 13 categories)

**memory** — MEM.1-MEM.3 n/a. MEM.4 holds (the history's two buffers by
the frame's parity, MEM.4's double buffer). MEM.5-MEM.8 n/a. MEM.9 P4.
MEM.10, MEM.11 n/a.

**copy-move** — COPY.1 holds. COPY.2 n/a (no move operations defined).
COPY.3 holds. COPY.4, COPY.5 hold. COPY.6 holds (every shared layout
asserted trivially copyable). COPY.7 P1. COPY.8 holds (`Read`, `Network`
returned by value). COPY.9 n/a.

**cache-layout** — CACHE.1 n/a. CACHE.2 holds (activations channels
innermost, as the matrices load). CACHE.3 holds. CACHE.4 holds (a pixel's
channels contiguous; the features as four images, each read whole).
CACHE.5 holds (asserted layouts). CACHE.6 n/a. CACHE.7 n/a. CACHE.8 left
to the counters (GPU.10).

**lifetime** — LIFE.1 holds. LIFE.4 holds (`memcpy` into vectors of
floats and halves). LIFE.2, LIFE.3, LIFE.5-LIFE.8 n/a.

**embedded** — EMB.1-EMB.8 n/a.

**concurrency** — CONC.1-CONC.8 n/a (no CPU threads; the GPU's atomics
relaxed and aggregated, P5).

**codegen** — GEN.1-GEN.8 n/a (no hints; the shader compiler's).

**simd** — SIMD.1-SIMD.8 n/a on the CPU; the GPU's vector access is
GPU.2's (P2).

**telemetry** — TLM.1-TLM.11 n/a (no instrumentation added; the timings
below come from a harness outside the repository).

**gpu** — GPU.1 P9. GPU.2 P2. GPU.3 P8. GPU.4 holds (conv3x3's inactive
rows are whole simdgroups). GPU.5 P6. GPU.6 holds (23 dispatches in the
frame's one encoder). GPU.7 holds (constants a ring per frame in flight).
GPU.8 P3. GPU.9 P4. GPU.10 holds (measured, below).

**gpu-dsa** — GDSA.1 n/a (no vendor library, the Phase 4 note says why).
GDSA.2 P5. GDSA.3 holds (the features' rays drawn as the light passes').
GDSA.4 holds (no spin-wait). GDSA.5 P5. GDSA.6 holds (bytes counted in the
headers). GDSA.7-GDSA.15 n/a. GDSA.16 P6. GDSA.17 P7. GDSA.18-GDSA.21 n/a.

**cpu-dsa** — CDSA.21 holds (`chunk_blocks`' rule stated and measured).
CDSA.32 P7. The rest n/a.

**wasm** — WASM.1-WASM.14 n/a.

## Counts

| | Findings | Fixed | Not valid | Convention |
|---|---|---|---|---|
| C++ Core Guidelines | 30 (G1-G30) | 21 (G1-G21) | 4 (G22-G25) | 5 (G26-G30) |
| Performance | 2 (P1, P2) | 2 (P1, P2) | 0 | 0 |

Seven more performance rules recorded as holding (P3-P9).

## Timings

GPU time a frame, median of 179 frames after 21 of warm-up, two frames in
flight, the marbles from t = 120 with time advancing a sixtieth of a
second a frame, each frame told the frame before it (as the window does);
the denoise pass alone as Phase 4 measured it, its inputs arbitrary. The
GPU idle before each run (utilization at most 5% for 6 s, no window).
The harness is outside the repository, from the earlier passes'. Three
builds, interleaved: **before** (139ed5d), **fixes** (every fix but P2)
and **after** (the fixes and P2, what is committed).

Runs that another load disturbed are dropped: once my own debug and
sanitize builds (CPU load lowers this package's GPU clock), and twice
GPU work from outside this session, which arrived mid-run after the idle
check passed (the frames of all three builds 10 to 45% slower together).
That left three clean runs of before and fixes, and two of after. Each
cell lists every kept run (ms):

| | Before | Fixes | After |
|---|---|---|---|
| restir_pt_denoised, 3456 × 2234 (light 1728 × 1117) | 77.02, 77.36, 77.01 | 76.76, 76.77, 76.77 | 76.69, 76.73 |
| restir_pt_denoised, 1920 × 1080 (light 960 × 540) | 23.63, 23.55, 23.58 | 23.53, 23.58, 23.54 | 23.43, 23.50 |
| the denoise pass, 3456 × 2234 | 26.28, 26.50, 26.34 | 26.05, 26.05, 26.05 | 25.97, 26.02 |
| the denoise pass, 1920 × 1080 | 7.44, 7.44, 7.44 | 7.36, 7.36, 7.37 | 7.31, 7.35 |
| restir_pt, 3456 × 2234 | 147.90, 147.88, 147.89 | 147.86, 148.05, 147.82 | as fixes |
| path, 3456 × 2234 | 29.99, 29.99, 29.99 | 29.97, 29.99, 30.00 | 29.98 |

- The fixes take 0.25 to 0.3 ms off the pass at the window's size (1%)
  and 0.08 ms at 1080p. Inferred, not taken apart: the convolution's and
  max pool's typed half4 indexing (G11) in place of cast byte pointers is
  the only change to how those kernels compute their addresses.
- P2 takes 0.03 to 0.05 ms more. Small, the same direction in every
  pair, nothing slower: kept.
- restir_pt and path are unchanged, as they should be: their kernels are
  the same, and the renderer's changes (G15, the comments) are a few
  host instructions a frame. After's restir_pt is fixes': between the two
  builds only the denoise kernels differ. After's one undisturbed-looking
  run measured 149.47, directly before a disturbed stretch; taken as its
  start (an inference) and not used.
- The window's frame is 76.7 ms, from 77.0.

## The Python

Reviewed for clarity, typing, unused names and bugs; no corpora. Nothing
that changes a number: the model's parameter names, the export's bytes,
the data set's plan (its SHA-256, pinned by
`tests/test_make_denoiser_dataset.py`) and `make_marbles.py`'s scenes
(the real marbles and all 40 seeds, byte for byte) are the same, and
`Denoiser.forward`, `frame`, `sequence_loss`, the augmentation, the crops,
the tone map, the regions and `input_frame` give the same bits on the CPU
before and after. `python3 tests/test_make_marbles.py`,
`tests/test_make_denoiser_dataset.py` and `uv run --frozen --offline python
-m unittest test_training` (13 tests) pass.

| Where | Finding | Verdict |
|---|---|---|
| training/report.py:30 | always added the OIDN column, so results from `evaluate.py` without `--oidn` failed with `KeyError` | fixed: the column only when the results have it |
| training/report.py:23 | no arguments, or one without `=`, failed with a traceback | fixed: the usage line, status 2 |
| training/tonemap.py:13, regions.py:21 | docstrings claimed checks in `tests/test_training.py` that do not exist (each was measured once, in the training note) | fixed: they say so |
| training/export.py, denoise_check.py:66, 152 | weights.bin's writing and its read-back by the layout written twice | fixed: `export.serialize()` and `read_back()`; the committed fixture's network regenerated byte for byte |
| training/data.py:105 | `_read_checked`, "private", imported by two other modules | fixed: `read_checked` |
| training/model.py, loss.py, train.py, evaluate.py, benchmark.py, denoise_check.py, scripts/*.py | unused locals and imports (`i`, `b`, `SPACINGS`, `attempt`, a shadowing `train`), a zero-filled buffer named `ones`, bare `dict` returns and missing annotations on public functions | fixed |
| scripts/make_denoiser_dataset.py:21 | the docstring calls t = 8 "the opening"; the plan labels it the flood, its strata being half-open | the docstring now says so; the label left as it is, since changing it would change the pinned plan |
| training/train.py:210-219 | a crash between the log's append and `last.pt`'s save would repeat that epoch's row on resume | reported, not changed: a narrow window, and the row's numbers are right |
| training/render_warmup.py:50 | `sha256` repeats `data.file_sha256` | not valid: importing `data` would load torch into a script that needs only numpy |
| training/evaluate.py:189 | reads radiance, albedo and normal again that `load_sequence` read | not valid: time only, in an offline evaluation |
