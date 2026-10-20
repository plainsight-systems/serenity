#pragma once

// Axis: Texture (shader half of textures/swirl.h).
//
// A marble core's twisting vanes at q, a point in the core's own
// coordinates, by core/textures/swirl.h's steps. No branch (GPU.4).
//
// Translated line for line from Metal's metal/textures/swirl.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/textures/swirl.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/textures/noise.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 swirl(serenity::textures::SwirlData data, float3 q) {
  // Step 1: q's angle about the axis, y, and its height.
  const float phi = atan2(q.z, q.x);
  // Step 2: its phase among the vanes, the noise read on the unit
  // cylinder at q's angle and height.
  const float3 on_cylinder = make_float3(cos(phi), q.y, sin(phi));
  const float s =
      float(data.vanes) * (phi * (0.5f * inverse_pi) + data.twist * q.y) +
      serenity::textures::swirl_waver *
          fbm(serenity::textures::swirl_scale * on_cylinder,
              serenity::textures::swirl_waver_octaves, data.seed);
  // Step 3: its color, a's bands and b's, their edges soft.
  const float v = abs(2.0f * fract(s) - 1.0f);
  const float w = smoothstep(0.5f - serenity::textures::swirl_edge,
                             0.5f + serenity::textures::swirl_edge, v);
  return mix(to_float3(data.a), to_float3(data.b), w);
}

}  // namespace shaders
}  // namespace serenity
