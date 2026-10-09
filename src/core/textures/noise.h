#pragma once

// Axis: Texture (noise: what the procedural textures build on).
//
// Gradient noise, decided here, in the core, so every backend's procedural
// textures draw the same pattern from the same seed (logical-overview.md,
// principle 10). A backend computes it in its shaders
// (metal/textures/noise.metal.h); no CPU copy computes it (MEMORY.md:
// correctness is shown on the GPU), and its tests check its properties
// there: continuous, 0 at every lattice point, within its bound, and a
// function of the point and the seed alone.
//
// noise(p, seed), for a point p in noise space and a seed, by these steps,
// which a backend's code carries by number:
//
//   Step 1  The lattice cell holding p: i = floor(p), an integer per axis,
//           and f = p - i, in [0, 1) per axis.
//   Step 2  Each of the cell's eight corners c = i + (0 or 1 per axis) is
//           hashed with the seed by pcg3d (Jarzynski and Olano, "Hash
//           Functions for GPU Rendering", JCGT 2020): h = pcg3d(uint3(c) +
//           uint3(seed)), the corner's integers taken as 32-bit unsigned,
//           wrapping. pcg3d of v, in 32-bit unsigned arithmetic:
//
//             v = v * 1664525 + 1013904223
//             v.x += v.y v.z;  v.y += v.z v.x;  v.z += v.x v.y
//             v ^= v >> 16
//             v.x += v.y v.z;  v.y += v.z v.x;  v.z += v.x v.y
//
//   Step 3  The corner's gradient: one of Perlin's twelve, the directions
//           to the midpoints of a cube's edges, (+-1, +-1, 0), (+-1, 0,
//           +-1), (0, +-1, +-1), the (h.x mod 12)th in the order of
//           noise_gradients below; its value at p, the gradient's dot product
//           with p - c.
//   Step 4  The eight values blended by f, axis by axis, with Perlin's
//           quintic fade 6t^5 - 15t^4 + 10t^3 (Perlin, "Improving Noise",
//           SIGGRAPH 2002): its first and second derivatives are 0 at the
//           corners, so the noise is smooth across cells.
//
// The noise is 0 at every lattice point, its mean over space 0, and its
// magnitude within noise_bound, which the tests hold it to.
//
// fbm(p, octaves, seed), fractional Brownian motion: the sum over k = 0 ..
// octaves - 1 of noise(2^k p, seed + k) / 2^k, each octave half the size and
// half the strength of the one before, and seeded apart, so octaves do not
// line up at the lattice points they share. Within noise_bound times 2 (1 -
// 2^-octaves).
//
// Cost, per noise: one pcg3d per corner, eight, each a dozen integer
// operations; eight dot products and seven blends.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define SERENITY_CONSTANT constant constexpr
#else
#define SERENITY_CONSTANT inline constexpr
#endif

namespace serenity {
namespace textures {

// Step 3's gradients, in the order h.x mod 12 picks them.
SERENITY_CONSTANT int noise_gradients[12][3] = {
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},  //
    {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},  //
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
};

// What |noise| never exceeds: 3D gradient noise with these gradients stays
// within about 1 (Perlin 2002); the tests hold it to this.
SERENITY_CONSTANT float noise_bound = 1.1f;

}  // namespace textures
}  // namespace serenity

#undef SERENITY_CONSTANT
