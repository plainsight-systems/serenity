# What the first medium costs a path

*2026-10-09 and 2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal
compiler 32023.864, `-std=metal4.0`. Release builds.*

## Question

The marbles brought the first participating medium (contract 12: absorbing
glass) and, with it, two things every path pays for whether or not it ever
enters glass: the medium the path is in, carried through the loop
(`src/metal/integrator/path.metal.h`, step 6), and each surface's
object-space point and interior, carried in `SurfaceInteraction`
(`src/core/contracts/surface_interaction.h`, 48 bytes to 64). What does a
scene with no medium pay for them?

## Method

The path tracer at 3456 x 2234 on the five-sphere marbles of a987908 (no
medium, no coat), nothing else on the GPU, the median frame time as in
2026-10-10-scene-block.md. Frame times only (GPU.10).

## Results

| Change | Cost |
|---|---|
| Storing `object_position` in `SurfaceInteraction` (48 to 64 bytes, live in registers through shading) | 0.21 ms of 14.8 |
| Carrying the medium through the path loop | 0.58 ms of 14.8 |
| Guarding the transmittance call, or marking air, to skip it | each under 0.07 ms saved |
| Measuring each stretch from the true point, and dimming each light sample by the medium (the review of df9d622) | 0.07–0.09 ms more |

The medium's cost is the state the loop keeps, not the branch: a path in
air asks no medium anything, and lanes diverge only where a SIMD group
straddles glass and air, a marble's few pixels (GPU.4).

The marbles (`scenes/marbles.toml`) load in 128 ms, their flights made in
parallel; a frame, path traced and tone mapped, is 11.2 ms (24e78a3).

## Decision

Kept: both are what the marbles need, and the guards that would skip the
medium's work in air saved too little to carry.
