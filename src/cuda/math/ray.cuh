#pragma once

// Axis: GPU backend (device code).
//
// A ray as the shaders carry it, in place of Metal's own bundle
// (metal::raytracing::ray, which CUDA has no counterpart of): an origin, a
// unit direction, and the open interval (min_distance, max_distance) its
// hits lie in (I.23), which a trace hands to OptiX as they are
// (acceleration/trace.cuh). The same four members under the same names, so
// the translated headers read them as Metal's did.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

struct Ray {
  float3 origin;
  float3 direction;
  float min_distance;
  float max_distance;
};

// Metal's ray(origin, direction, min_distance, max_distance).
__device__ inline Ray make_ray(float3 origin, float3 direction,
                               float min_distance, float max_distance) {
  return Ray{origin, direction, min_distance, max_distance};
}

}  // namespace shaders
}  // namespace serenity
