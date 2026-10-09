#pragma once

// Axis: Shape (shader half of shapes/box.h).
//
// The exact hit of a ray on a box about the origin, in object space, by
// slabs, and the outward normal of the face a point lies on there. The
// slabs need no unit direction: t is where the ray crosses each plane,
// whatever the direction's length (contracts/transform.h). A ray that starts
// inside the box hits it on the way out.

#include <metal_stdlib>

#include "core/shapes/box.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

// `direction` is unit length. On a hit, `t` is its distance.
inline bool intersect_box(serenity::shapes::BoxData box, float3 origin, float3 direction, float t_min, float t_max,
                          thread float& t) {
    // A component of exactly zero would divide to infinity, which Metal's
    // default (fast) math does not promise to carry; a tiny one gives the
    // same slabs.
    const float3 safe = metal::select(direction, float3(1e-20f), metal::abs(direction) < 1e-20f);
    const float3 inverse = 1.0f / safe;
    const float3 a = (to_float3(box.min) - origin) * inverse;
    const float3 b = (to_float3(box.max) - origin) * inverse;
    const float3 lower = metal::min(a, b);
    const float3 upper = metal::max(a, b);
    const float enter = metal::max(metal::max(lower.x, lower.y), lower.z);
    const float leave = metal::min(metal::min(upper.x, upper.y), upper.z);
    if (enter > leave) {
        return false;
    }
    if (enter > t_min && enter < t_max) {
        t = enter;
        return true;
    }
    if (leave > t_min && leave < t_max) {
        t = leave;
        return true;
    }
    return false;
}

// The face whose plane `point` is nearest, measured in units of the box's
// half size on that axis.
inline float3 box_normal(serenity::shapes::BoxData box, float3 point) {
    const float3 center = 0.5f * (to_float3(box.min) + to_float3(box.max));
    const float3 half_size = 0.5f * (to_float3(box.max) - to_float3(box.min));
    const float3 local = (point - center) / half_size;
    const float3 reach = metal::abs(local);
    if (reach.x >= reach.y && reach.x >= reach.z) {
        return float3(metal::sign(local.x), 0.0f, 0.0f);
    }
    if (reach.y >= reach.z) {
        return float3(0.0f, metal::sign(local.y), 0.0f);
    }
    return float3(0.0f, 0.0f, metal::sign(local.z));
}

}  // namespace shaders
}  // namespace serenity
