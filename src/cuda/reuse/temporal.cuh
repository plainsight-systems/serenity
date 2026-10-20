#pragma once

// Axis: Sample reuse (the temporal input's search order).
//
// Where ReSTIR PT's temporal reuse looks for the surface a pixel saw last
// frame (metal/passes/restir_pt/restir_pt.metal.h, step 5, previous_pixel):
// the pixel p its primary hit reprojects to, then, while none has passed
// the similarity test, up to 8 more, the pixels around p in ring order from
// a place the frame draws (RTXDI tries up to 9 pixels in a small
// neighborhood of the reprojection, the further ones at random). Not from a
// fixed place: then a pixel whose first try fails takes, frame after frame,
// the neighbor on the same side, and its history drifts one way across the
// image, which on a grazing floor measurably moved the image's mean
// (docs/research/2026-10-11-restir-temporal.md); a start the frame draws,
// the same for every pixel, turns that drift into a walk with no side (its
// number the frame's: GDSA.3).
//
// Pure functions and constants; depends on nothing.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// Pixels tried for the temporal input: the reprojected one and the
// ring_size around it.
inline constexpr uint temporal_tries = 9u;
inline constexpr uint ring_size = temporal_tries - 1u;

// The further pixels, around the reprojected one, in their fixed order
// (ES.27's C array: a shader constant table, as the noise tables): their
// x and y offsets, in device memory, which device code indexes at run time
// (CUDA's device code takes a namespace-scope constant of a scalar type
// alone; textures/noise.cuh).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): a device table
static constexpr __device__ int temporal_ring_x[ring_size] = {-1, 0,  1, -1,
                                                              1,  -1, 0, 1};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): a device table
static constexpr __device__ int temporal_ring_y[ring_size] = {-1, -1, -1, 0,
                                                              0,  1,  1,  1};

// Try t > 0's pixel: the ring's (t - 1 + start)th place around `p`.
__device__ inline int2 ring_neighbor(int2 p, uint t, uint start) {
  const uint place = (t - 1u + start) % ring_size;
  return p + make_int2(temporal_ring_x[place], temporal_ring_y[place]);
}

// Where the ring's tries start, from `u` in [0, 1): each place alike.
__device__ inline uint draw_ring_start(float u) {
  return min(uint(u * float(ring_size)), ring_size - 1u);
}

}  // namespace shaders
}  // namespace serenity
