#pragma once

// Axis: Shape (box).
//
// An axis-aligned box's data, shared with shaders. Its bounding box is
// itself, so the acceleration structure's box and the shape are the same;
// the shader half (metal/shapes/box.metal.h) still computes the exact entry
// point and the face's normal. The table, and the floor of the first scene,
// are boxes.

#include "core/contracts/float3.h"
#include "core/shapes/primitive.h"

namespace serenity {
namespace shapes {

struct BoxData {
    contracts::Float3 min;  // min < max on every axis
    uint32_t material;      // index into the material records
    contracts::Float3 max;
    uint32_t padding;
};

static_assert(sizeof(BoxData) == 32, "BoxData must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
inline Bounds bounds(const BoxData& box) {
    return Bounds{box.min, box.max};
}
#endif

}  // namespace shapes
}  // namespace serenity
