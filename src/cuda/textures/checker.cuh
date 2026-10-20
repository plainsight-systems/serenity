#pragma once

// Axis: Texture (shader half of textures/checker.h).
//
// The checker's color at point p, by the rule in textures/checker.h.
//
// Translated line for line from Metal's metal/textures/checker.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/textures/checker.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 checker(serenity::textures::CheckerData data,
                                 float3 point) {
  const float squares = floor(point.x / data.size) + floor(point.z / data.size);
  const bool even = fmod(abs(squares), 2.0f) < 0.5f;
  return even ? to_float3(data.a) : to_float3(data.b);
}

}  // namespace shaders
}  // namespace serenity
