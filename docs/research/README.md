# Research

Measurements and investigations, one note each. A note is dated and names the
machine, toolchain and build it ran on; its figures hold for those.

Newest first:

| Date | Note | What it found | Status |
|---|---|---|---|
| 2026-10-08 | [Metal math modes](2026-10-08-metal-math-modes.md) | CPU and GPU give equal bits only under `safe` and `precise` math with contraction off on both sides, for + - * /, sqrt and fma; not for subnormals, which the GPU flushes, nor for NaN encodings, which it canonicalizes | current |
