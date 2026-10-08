#pragma once

// Axis: Shape (sphere).
//
// A sphere's data, shared with shaders, and its bounding box for the
// acceleration structure. The exact hit and the normal are the shader half
// (metal/shapes/sphere.metal.h): the same kind, in two files, as a shader and
// its launcher are (change-axes.md).

#include "core/contracts/float3.h"
#include "core/shapes/primitive.h"

namespace serenity {
namespace shapes {

struct SphereData {
    contracts::Float3 center;
    float radius;       // greater than 0
    uint32_t material;  // index into the material records
    uint32_t padding[3];
};

static_assert(sizeof(SphereData) == 32, "SphereData must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// The box the acceleration structure holds for it: center plus and minus the
// radius on each axis.
inline Bounds bounds(const SphereData& sphere) {
    const contracts::Float3& c = sphere.center;
    const float r = sphere.radius;
    return Bounds{{c.x - r, c.y - r, c.z - r}, {c.x + r, c.y + r, c.z + r}};
}
#endif

}  // namespace shapes
}  // namespace serenity
