#pragma once

// Axis: Shape (box).
//
// An axis-aligned box in object space, its geometry its min and max corners,
// as the scene file gives them: a scene's boxes are placed by the identity
// transform, so the file's numbers are the box's, exactly. (A box held as a
// center and a half extent cannot keep them: the two, each rounded to a
// float, need not give back the corners; 100000000 to 100000008 would come
// back as 99999996 to 100000004.) Each box is a geometry of its own, boxes
// differing in shape and a scene having few. The table, and the floor of the
// first scene, are boxes. The shader half (metal/shapes/box.metal.h)
// computes the exact entry point and the face's normal.

#include "core/contracts/float3.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"

namespace serenity {
namespace shapes {

struct BoxData {
    contracts::Float3 min;  // min < max on every axis
    uint32_t padding0;
    contracts::Float3 max;
    uint32_t padding1;
};

static_assert(sizeof(BoxData) == 32, "BoxData must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// Its extent in object space: itself.
inline Bounds bounds(const BoxData& box) {
    return Bounds{box.min, box.max};
}

// Whether the box `transform` places meets or touches the world box
// `other`: their ranges overlap or meet on every axis. The transform is a
// translation and a uniform scale (contracts/transform.h), so the placed box is still
// axis-aligned.
bool box_touches(const BoxData& box, const contracts::Transform& transform, const Bounds& other);

// The distance from `point` to the box's surface, in double: outside, to its
// nearest point; inside, minus the distance to its nearest face.
double box_distance(const BoxData& box, const contracts::Transform& transform, contracts::Float3 point);
#endif

}  // namespace shapes
}  // namespace serenity
