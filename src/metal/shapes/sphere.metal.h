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
// cancel. The nearer of the two that lies within the ray's (min_distance,
// max_distance), so a ray that starts inside the sphere hits it on the way
// out.

#include <metal_raytracing>
#include <metal_stdlib>

#include "metal/shapes/crossing.metal.h"

namespace serenity {
namespace shaders {

// `r` in object space, its direction of any length. A crossing's t is the
// world's too (contracts/transform.h).
inline Crossing intersect_sphere(metal::raytracing::ray r) {
    const float a = metal::dot(r.direction, r.direction);
    const float b = metal::dot(r.origin, r.direction);
    const float3 nearest = r.origin - (b / a) * r.direction;
    const float discriminant = a * (1.0f - metal::dot(nearest, nearest));
    if (discriminant < 0.0f) {
        return no_crossing();
    }
    const float q = -(b + metal::copysign(metal::sqrt(discriminant), b));
    if (q == 0.0f) {
        return no_crossing();  // a ray from the sphere's surface, along it: both roots 0
    }
    const float c = metal::dot(r.origin, r.origin) - 1.0f;
    const float root_c = c / q;
    const float root_a = q / a;
    const float near = metal::min(root_c, root_a);
    const float far = metal::max(root_c, root_a);
    if (near > r.min_distance && near < r.max_distance) {
        return Crossing{true, near};
    }
    if (far > r.min_distance && far < r.max_distance) {
        return Crossing{true, far};
    }
    return no_crossing();
}

// The outward normal at `point`, in object space: the point itself, on the
// unit sphere.
inline float3 sphere_normal(float3 point) {
    return metal::normalize(point);
}

}  // namespace shaders
}  // namespace serenity
