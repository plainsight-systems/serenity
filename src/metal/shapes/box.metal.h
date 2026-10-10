#pragma once

// Axis: Shape (shader half of shapes/box.h).
//
// The exact hit of a ray on an axis-aligned box, in object space, by
// slabs, and the outward normal of the face a point lies on there. The
// slabs need no unit direction: t is where the ray crosses each plane,
// whatever the direction's length (contracts/transform.h). A ray that starts
// inside the box hits it on the way out.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/shapes/box.h"
#include "metal/device/layout.metal.h"
#include "metal/shapes/crossing.metal.h"

namespace serenity {
namespace shaders {

// The smallest direction component the slabs divide by. A component of
// exactly zero would divide to infinity, which Metal's default (fast) math
// does not promise to carry; one this small gives the same slabs, its
// planes' t beyond any scene.
constant constexpr float least_component = 1e-20f;

// `r` in object space, its direction of any length. A crossing's t is the
// world's too (contracts/transform.h).
inline Crossing intersect_box(serenity::shapes::BoxData box, metal::raytracing::ray r) {
    const float3 safe =
        metal::select(r.direction, float3(least_component), metal::abs(r.direction) < least_component);
    const float3 inverse = 1.0f / safe;
    const float3 a = (to_float3(box.min) - r.origin) * inverse;
    const float3 b = (to_float3(box.max) - r.origin) * inverse;
    const float3 lower = metal::min(a, b);
    const float3 upper = metal::max(a, b);
    const float enter = metal::max(metal::max(lower.x, lower.y), lower.z);
    const float leave = metal::min(metal::min(upper.x, upper.y), upper.z);
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
