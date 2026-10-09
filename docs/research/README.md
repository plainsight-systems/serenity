# Research

Measurements and investigations, one note each. A note is dated and names the
machine, toolchain and build it ran on; its figures hold for those.

Newest first:

| Date | Note | What it found | Status |
|---|---|---|---|
| 2026-10-09 | [The acceleration structure for moving shapes](2026-10-09-acceleration-structure.md) | on the M3 Max, an instance level over shader-tested boxes costs more per ray than rebuilding one level of placed boxes each frame (18.1 ms instanced, 8.2 ms one level, 7.8 ms before); triangle spheres in hardware 11.5 ms with no visible facets from 5,120 triangles | adopted: one level |
| 2026-10-08 | [Metal math modes](2026-10-08-metal-math-modes.md) | CPU and GPU give equal bits only under `safe` and `precise` math with contraction off on both sides, for + - * /, sqrt and fma; not for subnormals, which the GPU flushes, nor for NaN encodings, which it canonicalizes | facts stand; the contract it served was dropped 2026-10-08 |
