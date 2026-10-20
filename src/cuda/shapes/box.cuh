#pragma once

// Axis: Shape (shader half of shapes/box.h).
//
// The exact hit of a ray on an axis-aligned box, in object space, by
// slabs, the outward normal of the face a point lies on there, and a point
// found near a face put back on its plane. The slabs need no unit
// direction: t is where the ray crosses each plane, whatever the
// direction's length (contracts/transform.h). A ray that starts inside the
// box hits it on the way out.
//
// Translated line for line from Metal's metal/shapes/box.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/shapes/box.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/ray.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/shapes/crossing.cuh"

namespace serenity {
namespace shaders {

// The smallest direction component the slabs divide by. A component of
// exactly zero would divide to infinity, which Metal's default (fast) math
// does not promise to carry; one this small gives the same slabs, its
// planes' t beyond any scene. CUDA's IEEE arithmetic carries the infinity;
// the bound is kept so both backends cross the same slabs.
inline constexpr float least_component = 1e-20f;

// `r` in object space, its direction of any length. A crossing's t is the
// world's too (contracts/transform.h).
__device__ inline Crossing intersect_box(serenity::shapes::BoxData box, Ray r) {
  const float3 safe = select(r.direction, make_float3(least_component),
                             abs(r.direction) < least_component);
  const float3 inverse = 1.0f / safe;
  const float3 a = (to_float3(box.min) - r.origin) * inverse;
  const float3 b = (to_float3(box.max) - r.origin) * inverse;
  const float3 lower = min(a, b);
  const float3 upper = max(a, b);
  const float enter = max(max(lower.x, lower.y), lower.z);
  const float leave = min(min(upper.x, upper.y), upper.z);
  if (enter > leave) {
    return no_crossing();
  }
  if (enter > r.min_distance && enter < r.max_distance) {
    return Crossing{true, enter};
  }
  if (leave > r.min_distance && leave < r.max_distance) {
    return Crossing{true, leave};
  }
  return no_crossing();
}

// The outward normal, in object space, of the face whose plane `point` is
// nearest, measured in units of the box's half size on that axis.
__device__ inline float3 box_normal(serenity::shapes::BoxData box,
                                    float3 point) {
  const float3 center = 0.5f * (to_float3(box.min) + to_float3(box.max));
  const float3 half_size = 0.5f * (to_float3(box.max) - to_float3(box.min));
  const float3 local = (point - center) / half_size;
  const float3 reach = abs(local);
  if (reach.x >= reach.y && reach.x >= reach.z) {
    return make_float3(sign(local.x), 0.0f, 0.0f);
  }
  if (reach.y >= reach.z) {
    return make_float3(0.0f, sign(local.y), 0.0f);
  }
  return make_float3(0.0f, 0.0f, sign(local.z));
}

// `point`, found near the face whose outward normal is `normal`
// (box_normal()), on that face's plane: its coordinate on the face's axis
// the corner's, exactly; the other two as they are.
__device__ inline float3 box_surface_point(serenity::shapes::BoxData box,
                                           float3 point, float3 normal) {
  const float3 plane =
      select(to_float3(box.min), to_float3(box.max), normal > 0.0f);
  return select(point, plane, normal != 0.0f);
}

}  // namespace shaders
}  // namespace serenity
