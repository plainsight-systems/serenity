#pragma once

// Axis: Shape (shader half of shapes/sphere.h).
//
// The exact hit of a ray on a sphere, and the sphere's outward normal. The
// nearer of the two roots that lies in (t_min, t_max), so a ray that starts
// inside the sphere hits it on the way out.

#include <metal_stdlib>

#include "core/shapes/sphere.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

// `direction` is unit length. On a hit, `t` is its distance.
inline bool intersect_sphere(serenity::shapes::SphereData sphere, float3 origin, float3 direction, float t_min,
                             float t_max, thread float& t) {
    const float3 to_origin = origin - to_float3(sphere.center);
    const float b = metal::dot(to_origin, direction);
    const float c = metal::dot(to_origin, to_origin) - sphere.radius * sphere.radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = metal::sqrt(discriminant);
    const float near = -b - root;
    const float far = -b + root;
    if (near > t_min && near < t_max) {
        t = near;
        return true;
    }
    if (far > t_min && far < t_max) {
        t = far;
        return true;
    }
    return false;
}

inline float3 sphere_normal(serenity::shapes::SphereData sphere, float3 point) {
    return metal::normalize(point - to_float3(sphere.center));
}

}  // namespace shaders
}  // namespace serenity
