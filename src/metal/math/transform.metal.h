#pragma once

// Shared shader mathematics: contract 10's transforms (core/contracts/
// transform.h) applied in a shader. Each is a similarity, a rotation R, a
// uniform scale s and a translation T, as the 3 x 4 row-major matrix
// [s R | T]; so its inverse needs no general inversion: R^T / s, since
// (s R)^T (s R) = s^2 I. The Shape family carries hit points and normals
// between spaces with these (shapes/shapes.metal.h), and the Light family
// reads a sphere light's center and radius (lights/sphere_light.metal.h).

#include <metal_stdlib>

#include "core/contracts/transform.h"

namespace serenity {
namespace shaders {

inline float3 transform_row(serenity::contracts::Transform x, uint r) {
    return float3(x.m[r][0], x.m[r][1], x.m[r][2]);
}

// T: where the object's origin is in the world.
inline float3 transform_translation(serenity::contracts::Transform x) {
    return float3(x.m[0][3], x.m[1][3], x.m[2][3]);
}

// s: the length of the first column, any column's.
inline float transform_scale(serenity::contracts::Transform x) {
    return metal::length(float3(x.m[0][0], x.m[1][0], x.m[2][0]));
}

// (s R) v: a direction from object space into the world, scaled by s.
inline float3 transform_direction(serenity::contracts::Transform x, float3 v) {
    return float3(metal::dot(transform_row(x, 0), v), metal::dot(transform_row(x, 1), v),
                  metal::dot(transform_row(x, 2), v));
}

// R^T (p - T) / s: a world point in object space.
inline float3 transform_to_object(serenity::contracts::Transform x, float3 p) {
    const float3 q = p - transform_translation(x);
    const float3 rotated = transform_row(x, 0) * q.x + transform_row(x, 1) * q.y + transform_row(x, 2) * q.z;
    const float s = transform_scale(x);
    return rotated / (s * s);
}

// A world ray in object space: R^T (o - T) / s and R^T d / s, the
// direction not renormalized, so a parameter t means the same point in both
// (contracts/transform.h). One s^2 and two transposed products.
inline void transform_ray_to_object(serenity::contracts::Transform x, float3 origin, float3 direction,
                                    thread float3& object_origin, thread float3& object_direction) {
    const float3 r0 = transform_row(x, 0);
    const float3 r1 = transform_row(x, 1);
    const float3 r2 = transform_row(x, 2);
    const float inverse_s2 = 1.0f / metal::dot(float3(x.m[0][0], x.m[1][0], x.m[2][0]),
                                               float3(x.m[0][0], x.m[1][0], x.m[2][0]));
    const float3 q = origin - transform_translation(x);
    object_origin = (r0 * q.x + r1 * q.y + r2 * q.z) * inverse_s2;
    object_direction = (r0 * direction.x + r1 * direction.y + r2 * direction.z) * inverse_s2;
}

}  // namespace shaders
}  // namespace serenity
