#pragma once

// Shared shader mathematics: contract 10's transforms (core/contracts/
// transform.h) applied in a shader. Each is a similarity, a rotation R, a
// uniform scale s and a translation T, as the 3 x 4 row-major matrix
// [s R | T]; so its inverse needs no general inversion: R^T / s, since
// (s R)^T (s R) = s^2 I. The Shape family carries hit points and normals
// between spaces with these (shapes/shapes.cuh), and the Light family
// reads a sphere light's center and radius (lights/sphere_light.cuh).
//
// Translated line for line from Metal's metal/math/transform.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/transform.h"
#include "cuda/math/ray.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 transform_row(serenity::contracts::Transform x,
                                       uint r) {
  return make_float3(x.m[r][0], x.m[r][1], x.m[r][2]);
}

// Column c of s R: s times the unit vector R carries axis c to.
__device__ inline float3 transform_column(serenity::contracts::Transform x,
                                          uint c) {
  return make_float3(x.m[0][c], x.m[1][c], x.m[2][c]);
}

// T: where the object's origin is in the world.
__device__ inline float3 transform_translation(
    serenity::contracts::Transform x) {
  return make_float3(x.m[0][3], x.m[1][3], x.m[2][3]);
}

// s: the length of the first column, any column's.
__device__ inline float transform_scale(serenity::contracts::Transform x) {
  return length(transform_column(x, 0));
}

// 1 / s^2, from the first column's squared length: no square root.
__device__ inline float transform_inverse_scale2(
    serenity::contracts::Transform x) {
  const float3 column = transform_column(x, 0);
  return 1.0f / dot(column, column);
}

// (s R) v: a direction from object space into the world, scaled by s.
__device__ inline float3 transform_direction(serenity::contracts::Transform x,
                                             float3 v) {
  return make_float3(dot(transform_row(x, 0), v), dot(transform_row(x, 1), v),
                     dot(transform_row(x, 2), v));
}

// (s R)^T v: the transposed product; divided by s^2, a world vector carried
// into object space.
__device__ inline float3 transform_transposed(serenity::contracts::Transform x,
                                              float3 v) {
  return transform_row(x, 0) * v.x + transform_row(x, 1) * v.y +
         transform_row(x, 2) * v.z;
}

// R^T (p - T) / s: a world point in object space.
__device__ inline float3 transform_to_object(serenity::contracts::Transform x,
                                             float3 p) {
  return transform_transposed(x, p - transform_translation(x)) *
         transform_inverse_scale2(x);
}

// s R p + T: an object-space point in the world, transform_to_object()'s
// inverse.
__device__ inline float3 transform_to_world(serenity::contracts::Transform x,
                                            float3 p) {
  return transform_direction(x, p) + transform_translation(x);
}

// A world ray in object space: R^T (o - T) / s and R^T d / s, the
// direction not renormalized, so a parameter t means the same point in both
// (contracts/transform.h), and the ray's bounds carry over as they are. One
// 1 / s^2 and two transposed products.
__device__ inline Ray transform_ray_to_object(serenity::contracts::Transform x,
                                              Ray world) {
  const float inverse_s2 = transform_inverse_scale2(x);
  return make_ray(
      transform_transposed(x, world.origin - transform_translation(x)) *
          inverse_s2,
      transform_transposed(x, world.direction) * inverse_s2, world.min_distance,
      world.max_distance);
}

}  // namespace shaders
}  // namespace serenity
