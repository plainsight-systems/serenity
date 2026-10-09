#pragma once

// Axis: Shape (shader half of shapes/sphere.h).
//
// The exact hit of a ray on the unit sphere, in object space, and its
// outward normal there, the point itself. The ray's direction is not unit
// length (contracts/transform.h: it is the world's carried into object
// space, scaled by 1 / radius), so the roots of |o + t d|^2 = 1 are found
// with a = d.d kept, b = o.d, c = o.o - 1.
//
// In the stable form of Haines et al. (Ray Tracing Gems, ch. 7), since a
// small sphere seen from afar is the scene's whole point: the discriminant
// b^2 - a c, two numbers near (|o| |d|)^2 subtracted, loses nearly every
// digit for a firefly 3 cm wide 5 m off (a hit 2e-5 short, in float); it is
// a (1 - |l|^2) instead, l = o - (b / a) d the vector from the center to the
// ray's nearest point. And the roots are c / q and q / a, q = -(b + sign(b)
// sqrt(disc)), which adds numbers of one sign where -b + sqrt(disc) would
// cancel. The nearer of the two that lies in (t_min, t_max), so a ray that
// starts inside the sphere hits it on the way out.

#include <metal_stdlib>

#include "core/shapes/sphere.h"

namespace serenity {
namespace shaders {

// `origin` and `direction` in object space, `direction` of any length. On a
// hit, `t` is its parameter, the world's too (contracts/transform.h).
inline bool intersect_sphere(float3 origin, float3 direction, float t_min, float t_max, thread float& t) {
    const float a = metal::dot(direction, direction);
    const float b = metal::dot(origin, direction);
    const float3 nearest = origin - (b / a) * direction;
    const float discriminant = a * (1.0f - metal::dot(nearest, nearest));
    if (discriminant < 0.0f) {
        return false;
    }
    const float q = -(b + metal::copysign(metal::sqrt(discriminant), b));
    if (q == 0.0f) {
        return false;  // a ray from the sphere's surface, along it: both roots 0
    }
    const float c = metal::dot(origin, origin) - 1.0f;
    const float one = c / q;
    const float other = q / a;
    const float near = metal::min(one, other);
    const float far = metal::max(one, other);
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

// The outward normal at `point`, in object space: the point itself, on the
// unit sphere.
inline float3 sphere_normal(float3 point) {
    return metal::normalize(point);
}

}  // namespace shaders
}  // namespace serenity
