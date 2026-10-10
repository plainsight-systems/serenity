# What limits the path kernel: GPU counters

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2 (Instruments 26),
Metal compiler 32023.864, `-std=metal4.0`. Release build at 3e107d3.*

## Question

Two redesigns taken from other renderers' advice (2026-10-10-scene-block.md,
2026-10-10-primitive-data.md) were judged by frame time alone. Where does
the path tracer's frame actually go (GPU.10)?

## Method

The path tracer (`graphs/path.toml`) on the five-sphere marbles at 3456 x
2234, 200 frames back to back, the GPU otherwise idle. Instruments' Metal
GPU Counters instrument with the Performance Limiters counter set, saved as
a user template ("Serenity GPU Counters"; the instrument's default counter
set is None, which records nothing). Recorded with `xctrace record
--all-processes` started before the run: launching the run from `xctrace`
crashes it (SIGSEGV in GPUPlugin) once counters are on. Each counter's mean
is taken over the 3.03 s in which the compute kernel occupied the GPU
(Kernel Occupancy above 50%, then the whole window), each counter group on
its own sampling clock, about 115,000 samples a counter. The run's median
frame was 15.16 ms under the recorder, 15.0 without.

The texture, bandwidth, MMU and last-level-cache counters of the set
recorded no samples in the window. The same recording of
`scenes/marbles.toml` failed twice to save ("Document Missing Template
Error"), so these figures are for the five spheres alone.

## Results

| Counter | Mean |
|---|---|
| RT Unit Active | 87.9% |
| Kernel Occupancy | 24.8% |
| Occupancy Manager Target | 30.7% |
| L1 Eviction Rate | 85.1% |
| L1 Total Residency | 75.1% |
| L1 RT Scratch Residency | 45.3% |
| L1 Register Residency | 20.2% |
| L1 Buffer Residency | 8.1% |
| Buffer L1 Miss Rate | 24.9% |
| Compute SIMD Groups Inflight (per core) | 23.8 |
| Instruction Throughput Limiter / Utilization | 33.2% / 9.2% |
| ALU Utilization | 13.0% |
| F32 Limiter / Utilization | 8.6% / 7.6% |
| Integer and Conditional Limiter / Utilization | 16.4% / 14.6% |
| Control Flow Limiter / Utilization | 9.9% / 7.6% |
| L1 Cache Limiter | 6.7% |
| Compute Shader Launch Limiter | 67.6% |

## Findings

Facts, from the counters:

- The kernel is not limited by arithmetic: ALU utilization 13%, every ALU
  sub-block's limiter under 17%.
- The ray tracing unit is active 88% of the time.
- Occupancy is 25%, and the occupancy manager's own target is 31%. Apple's
  description of L1 Eviction Rate: "The more this number approaches 100%
  the more that the Occupancy Manager will attempt to reduce simdgroup
  occupancy." It is 85%.
- What fills L1 is per-thread private state: ray tracing scratch 45% and
  registers 20%, against 8% for buffers, the scene's data.

Inference: the frame is bound by ray traversal at low occupancy. Each
thread keeps its intersection queries' scratch and the path's registers
(the throughput, the radiance found, the surface interaction, the BSDF, the
medium, the sampler) live across every query; that state evicts L1, so
Apple's occupancy manager runs a quarter of the threads it could, too few
to hide the RT unit's and memory's latency. This agrees with the two
spikes: the change that cut live pointers (views built at their use) and
the one that moved the scene's data into the constant address space helped,
and the one that read more per candidate from device memory (primitive
data) did not; the scene's data itself is a small part of L1.

## Decision

The lever is the state live across a ray query, not the scene's data:
fewer registers and less ray tracing scratch per thread, so more threads
run. The candidates, to be spiked and measured against these counters:

- Split the path kernel at its ray queries (wavefront path tracing: Laine,
  Karras and Aila, "Megakernels Considered Harmful", HPG 2013): kernels
  that only trace, holding a ray and a hit, between kernels that shade.
- Within the one kernel, keep one intersection query live at a time and
  less shading state across it.

Open: what each intersection query costs in scratch. The marbles' counters,
recorded after the guideline sweep with a shorter recording window (11 s
saves where 25 s did not), are in 2026-10-10-corpus-sweep.md: the same
picture, stronger.
