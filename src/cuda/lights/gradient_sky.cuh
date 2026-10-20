#pragma once

// Axis: Light (shader half of lights/gradient_sky.h).
//
// The radiance arriving from direction d, by the blend lights/gradient_sky.h
// states: smoothstep clamps d.y, so below the horizon it is the horizon.
//
// Translated line for line from Metal's metal/lights/gradient_sky.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/lights/gradient_sky.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 gradient_sky(serenity::lights::GradientSkyData sky,
                                      float3 direction) {
  const float blend = smoothstep(0.0f, 1.0f, direction.y);
  return to_float3(sky.horizon) +
         (to_float3(sky.zenith) - to_float3(sky.horizon)) * blend;
}

}  // namespace shaders
}  // namespace serenity
