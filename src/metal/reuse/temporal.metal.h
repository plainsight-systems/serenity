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

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// Pixels tried for the temporal input: the reprojected one and the
// ring_size around it.
constant constexpr uint temporal_tries = 9u;
constant constexpr uint ring_size = temporal_tries - 1u;

// The further pixels, around the reprojected one, in their fixed order
// (ES.27's C array: a shader constant table, as the noise tables).
constant constexpr int2 temporal_ring[ring_size] = {
    int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0),
    int2(1, 0),   int2(-1, 1), int2(0, 1),  int2(1, 1)};

// Try t > 0's pixel: the ring's (t - 1 + start)th place around `p`.
inline int2 ring_neighbor(int2 p, uint t, uint start) {
  return p + temporal_ring[(t - 1u + start) % ring_size];
}

// Where the ring's tries start, from `u` in [0, 1): each place alike.
inline uint draw_ring_start(float u) {
  return metal::min(uint(u * float(ring_size)), ring_size - 1u);
}

}  // namespace shaders
}  // namespace serenity
