# Research

Measurements and investigations, one note each. A note is dated and names the
machine, toolchain and build it ran on; its figures hold for those.

Newest first:

| Date | Note | What it found | Status |
|---|---|---|---|
| 2026-10-10 | [Shapes' hot data as the acceleration structure's primitive data](2026-10-10-primitive-data.md) | one 96-byte record per shape in the structure, read with `get_candidate_primitive_data()` (Apple's WWDC22 10105 advice), is 0.2–0.3 ms slower than three small tables in the constant address space; building it in costs nothing | not adopted |
| 2026-10-10 | [Binding the scene: one block, against Apple, Falcor and Unreal](2026-10-10-scene-block.md) | Apple's bindless scene, Falcor and Unreal on Metal all reach the scene through one block of addresses; read as `device`, the block cost 5% over per-slot bindings, as `constant` (Apple's example) under 2% (15.07 ms against 14.77); binding the hottest arrays directly (Falcor's `[root]`) and holding the sky by value saved nothing on Metal | adopted: the block, `constant`; per-primitive data open |
| 2026-10-09 | [What the first medium costs a path](2026-10-09-medium-cost.md) | in a scene with no medium, carrying the medium costs 0.58 ms of 14.8, the object-space point 0.21; guards to skip the medium in air save under 0.07 | kept |
| 2026-10-09 | [The acceleration structure for moving shapes](2026-10-09-acceleration-structure.md) | on the M3 Max, an instance level over shader-tested boxes costs more per ray than rebuilding one level of placed boxes each frame (18.1 ms instanced, 8.2 ms one level, 7.8 ms before); triangle spheres in hardware 11.5 ms with no visible facets from 5,120 triangles | adopted: one level |
| 2026-10-08 | [Metal math modes](2026-10-08-metal-math-modes.md) | CPU and GPU give equal bits only under `safe` and `precise` math with contraction off on both sides, for + - * /, sqrt and fma; not for subnormals, which the GPU flushes, nor for NaN encodings, which it canonicalizes | facts stand; the contract it served was dropped 2026-10-08 |
