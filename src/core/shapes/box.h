#pragma once

// Axis: Shape (box).
//
// A box about its origin, in object space: its geometry is its half
// extent on each axis, and its shape's transform (transform.h) places its
// center in the world. Each box is a geometry of its own, with its own
// acceleration structure of one box, built once (metal/acceleration/
// scene_acceleration.h): boxes differ in shape, and a scene has few. The
// table, and the floor of the first scene, are boxes. The shader half
// (metal/shapes/box.metal.h) computes the exact entry point and the face's
// normal.

#include "core/contracts/float3.h"
#include "core/shapes/primitive.h"
#include "core/shapes/transform.h"

namespace serenity {
namespace shapes {

struct BoxData {
    contracts::Float3 half_extent;  // greater than 0 on every axis
    uint32_t padding;
};

static_assert(sizeof(BoxData) == 16, "BoxData must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// Its extent in object space: -half_extent to half_extent.
inline Bounds bounds(const BoxData& box) {
    const contracts::Float3& h = box.half_extent;
    return Bounds{{-h.x, -h.y, -h.z}, {h.x, h.y, h.z}};
}

// Whether the box `transform` places meets or touches the world box
// `other`: their ranges overlap or meet on every axis. The transform is a
// translation and a uniform scale (transform.h), so the placed box is still
// axis-aligned.
bool box_touches(const BoxData& box, const Transform& transform, const Bounds& other);
#endif

}  // namespace shapes
}  // namespace serenity
