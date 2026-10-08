# Metal math modes and CPU-GPU bit equality

*2026-10-08. Apple M3 Max (Apple9, `applegpu_g15s`), macOS 26.6.2, Xcode 26.2
(17C52), Metal compiler 32023.864, Apple clang 17.0.0 (clang-1700.6.3.2).
Shader language `-std=metal3.2`.*

## Question

Can one source, compiled once as C++ and once as a Metal shader, give the
same bits on the CPU and the GPU? Tests that check GPU math against a CPU
computation of the same function need that, and it decides which flags every
shader is built with.

## Method

Two runs.

1. **Prototype** (scratch code, not committed). 2^20 triples (a, b, c),
   uniform in [-100, 100] from libc++'s `std::uniform_real_distribution` over
   `std::mt19937` seeded 1. Five expressions in one header included by both
   sides: `a * b + c`, `fma(a, b, c)`, `a / (b + c)`, `sqrt(a * a + b * b)`,
   `(a + b) - a + c`, then `exp(0.05 a) + sin(b)` in place of the last. The
   GPU compiled under each math mode; the C++ at `-O2` with
   `-ffp-contract=on` and `off`, and at `-O0`.
2. **The committed test**, `tests/gpu/portable_math_test.cpp`, run with
   `make test`. About 2^20 triples: special values, random bit patterns
   (every class of float), and random normals with exponents in [-20, 20],
   all from `std::mt19937`'s raw output. Seven operations
   (`tests/gpu/kernels/portable_math_probe.h`). Built twice: with the pinned
   flags, and with Metal's fast math.

## Results

**Prototype**, triples of 1,048,576 whose GPU result differs from the CPU's:

| GPU flags | CPU contraction | `a*b+c` | `fma` | `a/(b+c)` | `sqrt` | `(a+b)-a+c` |
|---|---|---|---|---|---|---|
| default (fast) | on | 0 | 0 | 286,342 | 345,440 | 201,437 |
| default (fast) | off | 268,197 | 0 | 286,107 | 331,624 | 55,078 |
| `relaxed` | on | 0 | 0 | 286,342 | 345,440 | 201,437 |
| `safe` | on | 0 | 0 | 0 | 292,317 | 0 |
| `safe` | off | 268,197 | 0 | 0 | 331,760 | 0 |
| `safe`, `precise` | on | 0 | 0 | 0 | 0 | 0 |
| `safe`, `precise` | off | 268,197 | 0 | 0 | 90,375 | 0 |
| `safe`, `precise`, `-ffp-contract=off` | on | 272,481 | 0 | 0 | 90,229 | 0 |
| `safe`, `precise`, `-ffp-contract=off` | off | **0** | **0** | **0** | **0** | **0** |

`safe` is `-fmetal-math-mode=safe`; `precise` is
`-fmetal-math-fp32-functions=precise`. With the last row's flags,
`exp(0.05 a) + sin(b)` differed in 414,760 triples (40%), the same at `-O0`.

**Committed test**, with the pinned flags (the last row above): no
differences in any of the seven operations, among the triples where no input,
rounded intermediate or result is subnormal, except in NaN results. Those are
NaN on both sides but differ in encoding: the GPU returned every one of
32,764 NaN results as `0x7fc00000`, one encoding; the CPU's took 6,161, since
it keeps the sign and payload of a NaN input. (Found by the independent
review of the commit that added the test, which first compared NaNs by class
alone.) Outside that domain, from 414 of
4,203 triples (addition) to 24,483 of 26,397 (division) differ. The first
mismatch of each operation, the only ones inspected, is a subnormal input the
GPU treated as zero or a subnormal result it returned as zero; that every
mismatch is one was not checked. Adding
`-fdenormal-fp-math=ieee` (accepted by the compiler without a warning)
changed nothing.

With Metal's fast math, inside the domain: division differs in 259,931 of
1,022,189 triples, sqrt in 192,047 of 996,662; addition, subtraction,
multiplication, `fma` and `a * b + c` in none. Fast math did not contract
`a * b + c` in this kernel, though it did in the prototype's.

## What it means

- Equal bits are reachable, NaN aside, and only under all of: `safe`, `precise`,
  contraction off on both sides, and IEEE-exact operations (+ - * /, sqrt,
  fma). This is the contract in `src/core/portable_math.h`, and the flags are
  pinned in `cmake/MetalLibrary.cmake`.
- `safe` alone still lets the compiler contract: in the prototype the GPU
  fused `a * b + c` under `safe`, matching the CPU only when the CPU fused
  too. Whether a compiler contracts depends on the shape of the code (the two
  runs disagree under fast math), so contraction is turned off rather than
  relied on, and `fma()` is called where fusion is wanted.
- `exp` and `sin` are not bit-portable even as `precise`. Math built on them
  needs its own arithmetic from the exact operations, or a tolerance.
- The GPU canonicalizes NaN. NaN results are compared by class, and the
  GPU's are checked to be its one encoding; a NaN's sign and payload are
  outside the contract.
- The GPU flushes subnormal floats to zero. The contract is narrowed to
  exclude them rather than making the CPU flush too, which would mean a
  thread-wide floating-point mode.

## Not measured

- What `safe` and `precise` cost in shader speed. They apply to every shader
  built with the pinned flags; the first rendering kernels should measure the
  difference where it might matter (transcendental-heavy shading, sampling).
- Half precision, integer conversions, and the remaining math functions.
- Whether the flush to zero holds on other Apple GPU families.
