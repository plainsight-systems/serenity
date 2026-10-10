# Shapes' hot data as the acceleration structure's primitive data

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864, `-std=metal4.0`. Release builds.*

## Question

Apple's ray tracing guidance (WWDC22 10105) is to store each primitive's
data in the acceleration structure and read it with one load,
`get_candidate_primitive_data()`, instead of chains of buffer lookups in the
intersection code: 10 to 16% in Apple's test. Unreal's GPUScene has the same
shape, one flat record per instance. Serenity's intersection loops
(`src/metal/acceleration/trace.metal.h`, both the camera ray's and every
shadow ray's) read, per candidate box, the shape's record, its transform,
and for a box its geometry, three arrays. Does one record in the structure
pay?

## Method

A throwaway spike on f582d42, in a worktree, not committed:

- A 96-byte record per shape: its transform (48), its `ShapeRecord` (16)
  and, for a box, its `BoxData` (32).
- One copy per frame in flight, set as the bounding-box geometry's
  `primitiveDataBuffer` (Metal 4's `MTL4AccelerationStructureGeometryDescriptor`).
  The moving shapes' transforms are written into it each frame, before the
  structure's rebuild copies it in.
- Both loops read it with `query.get_candidate_primitive_data()` in place of
  the three arrays. Metal's `intersection_query<>` needed no tag for this.

The GPU suite passed under Metal's validation layers, frame images
unchanged. Timing as in 2026-10-10-scene-block.md: the path tracer at 3456
x 2234, the median of frames 21 to 199. Main and spike were run alternately,
three rounds each, to rule out drift.

## Results

| Variant | Five spheres | Marbles |
|---|---|---|
| Main, the three arrays in the constant address space (f582d42) | 14.94–15.05 ms | 11.16–11.18 ms |
| Spike: the 96-byte record, read in both loops | 15.21–15.33 | 11.34–11.39 |
| Spike: the record built into the structure but not read | 14.95–14.99 | 11.18 |
| Spike: a 64-byte record, the box still from its array | 15.51–15.52 | 11.53 |

## Findings

- Building the record into the structure costs nothing measurable: the
  rebuild each frame copies 96 bytes a shape at no cost visible in a frame.
- Reading it costs 0.2–0.3 ms a frame more than the three arrays. Those
  arrays are a few hundred records at most, in the constant address space
  (2026-10-10-scene-block.md), read by every thread; the primitive data is
  read from the structure's own memory. Apple's 10–16% replaced reads of
  large per-triangle buffers (texture coordinates, material indices) and a
  texture sample, not small constant tables.
- A smaller record that leaves the box in its array is slower still: the
  box's read then waits on the record's.

## Decision

Not adopted. The shapes' data stays in the scene's block. Primitive data
becomes worth measuring again when the hot data per candidate grows past
what the constant address space serves well, such as triangle meshes with
per-triangle attributes. Without hardware counters (GPU.10), the frame's
time is not yet attributed to the intersection loops, shading or memory;
that attribution comes before any further redesign.
