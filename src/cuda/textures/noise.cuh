#pragma once

// Axis: Texture (shader half of textures/noise.h).
//
// Gradient noise and its octaves, by core/textures/noise.h's steps.
//
// Translated line for line from Metal's metal/textures/noise.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/textures/noise.h"
#include "cuda/math/hash.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// noise.h, step 3: as many gradients as its table holds.
inline constexpr uint gradient_count =
    sizeof(serenity::textures::noise_gradients) /
    sizeof(serenity::textures::noise_gradients[0]);

// The core's table of gradients, as device code reads it: CUDA lets device
// code index a host constant only in a constant expression (the CUDA
// Programming Guide's rules for constexpr variables), so the table is
// copied, at compile time, from the core's own into a constant of device
// memory, the same numbers from the one place they are written (ES.3).
struct NoiseGradients {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): the core's shape
  int g[gradient_count][3];
};

constexpr NoiseGradients noise_gradients_copy() {
  NoiseGradients out{};
  for (uint i = 0; i < gradient_count; ++i) {
    for (uint axis = 0; axis < 3; ++axis) {
      out.g[i][axis] = serenity::textures::noise_gradients[i][axis];
    }
  }
  return out;
}

static constexpr __device__ NoiseGradients device_noise_gradients =
    noise_gradients_copy();

// Step 3: corner c's gradient, dotted with the offset to the point; step
// 2's hash is pcg3d (math/hash.cuh).
__device__ inline float corner_value(int3 c, float3 offset, uint seed) {
  const uint3 h = pcg3d(as_uint3(c) + make_uint3(seed));
  const int* g = device_noise_gradients.g[h.x % gradient_count];
  return float(g[0]) * offset.x + float(g[1]) * offset.y +
         float(g[2]) * offset.z;
}

__device__ inline float gradient_noise(float3 p, uint seed) {
  // Step 1. The cell's corner as 32-bit integers: within their range for
  // every point a texture is read at (ES.46), as each texture's own
  // header bounds it (core/textures/wood.h: at most some 6.7e8; the
  // swirl's, on the unit cylinder, at most a few).
  const float3 cell = floor(p);
  const int3 i = make_int3(cell);
  const float3 f = p - cell;
  // Steps 2 and 3, at the eight corners.
  const float n000 = corner_value(i, f, seed);
  const float n100 =
      corner_value(i + make_int3(1, 0, 0), f - make_float3(1, 0, 0), seed);
  const float n010 =
      corner_value(i + make_int3(0, 1, 0), f - make_float3(0, 1, 0), seed);
  const float n110 =
      corner_value(i + make_int3(1, 1, 0), f - make_float3(1, 1, 0), seed);
  const float n001 =
      corner_value(i + make_int3(0, 0, 1), f - make_float3(0, 0, 1), seed);
  const float n101 =
      corner_value(i + make_int3(1, 0, 1), f - make_float3(1, 0, 1), seed);
  const float n011 =
      corner_value(i + make_int3(0, 1, 1), f - make_float3(0, 1, 1), seed);
  const float n111 =
      corner_value(i + make_int3(1, 1, 1), f - make_float3(1, 1, 1), seed);
  // Step 4: the quintic fade.
  const float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
  const float x00 = mix(n000, n100, u.x);
  const float x10 = mix(n010, n110, u.x);
  const float x01 = mix(n001, n101, u.x);
  const float x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

__device__ inline float fbm(float3 p, uint octaves, uint seed) {
  float sum = 0.0f;
  float scale = 1.0f;
  float strength = 1.0f;
  for (uint k = 0; k < octaves; ++k) {
    sum += strength * gradient_noise(p * scale, seed + k);
    scale *= 2.0f;
    strength *= 0.5f;
  }
  return sum;
}

}  // namespace shaders
}  // namespace serenity
