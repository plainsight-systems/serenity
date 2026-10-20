#pragma once

// Axis: Medium (shader half of media/absorbing.h).
//
// Beer and Lambert: a stretch t of an absorbing medium keeps exp(-absorption
// t) of the light, per channel.
//
// Translated line for line from Metal's metal/media/absorbing.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/media/absorbing.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 absorbing_transmittance(
    serenity::media::AbsorbingData medium, float t) {
  return exp(-to_float3(medium.absorption) * t);
}

}  // namespace shaders
}  // namespace serenity
