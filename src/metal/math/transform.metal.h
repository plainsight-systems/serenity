#pragma once

// Shared shader mathematics: contract 10's transforms (core/contracts/
// transform.h) applied in a shader. Each is a similarity, a rotation R, a
// uniform scale s and a translation T, as the 3 x 4 row-major matrix
// [s R | T]; so its inverse needs no general inversion: R^T / s, since
// (s R)^T (s R) = s^2 I. The Shape family carries hit points and normals
// between spaces with these (shapes/shapes.metal.h), and the Light family
// reads a sphere light's center and radius (lights/sphere_light.metal.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"

namespace serenity {
namespace shaders {

inline float3 transform_row(serenity::contracts::Transform x, uint r) {
    return float3(x.m[r][0], x.m[r][1], x.m[r][2]);
}

// Column c of s R: s times the unit vector R carries axis c to.
inline float3 transform_column(serenity::contracts::Transform x, uint c) {
    return float3(x.m[0][c], x.m[1][c], x.m[2][c]);
}

// T: where the object's origin is in the world.
inline float3 transform_translation(serenity::contracts::Transform x) {
    return float3(x.m[0][3], x.m[1][3], x.m[2][3]);
}

// s: the length of the first column, any column's.
inline float transform_scale(serenity::contracts::Transform x) {
    return metal::length(transform_column(x, 0));
}

// 1 / s^2, from the first column's squared length: no square root.
inline float transform_inverse_scale2(serenity::contracts::Transform x) {
    const float3 column = transform_column(x, 0);
    return 1.0f / metal::dot(column, column);
}

// (s R) v: a direction from object space into the world, scaled by s.
inline float3 transform_direction(serenity::contracts::Transform x, float3 v) {
    return float3(metal::dot(transform_row(x, 0), v), metal::dot(transform_row(x, 1), v),
                  metal::dot(transform_row(x, 2), v));
}

// (s R)^T v: the transposed product; divided by s^2, a world vector carried
// into object space.
inline float3 transform_transposed(serenity::contracts::Transform x, float3 v) {
    return transform_row(x, 0) * v.x + transform_row(x, 1) * v.y + transform_row(x, 2) * v.z;
}

// R^T (p - T) / s: a world point in object space.
inline float3 transform_to_object(serenity::contracts::Transform x, float3 p) {
    return transform_transposed(x, p - transform_translation(x)) * transform_inverse_scale2(x);
}

// A world ray in object space: R^T (o - T) / s and R^T d / s, the
// direction not renormalized, so a parameter t means the same point in both
// (contracts/transform.h), and the ray's bounds carry over as they are. One
// 1 / s^2 and two transposed products.
inline metal::raytracing::ray transform_ray_to_object(serenity::contracts::Transform x,
                                                      metal::raytracing::ray world) {
    const float inverse_s2 = transform_inverse_scale2(x);
    return metal::raytracing::ray(transform_transposed(x, world.origin - transform_translation(x)) * inverse_s2,
                                  transform_transposed(x, world.direction) * inverse_s2, world.min_distance,
                                  world.max_distance);
}

}  // namespace shaders
}  // namespace serenity
