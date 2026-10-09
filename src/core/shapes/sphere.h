#pragma once

// Axis: Shape (sphere).
//
// The unit sphere: center at the origin, radius 1, in object space. It has
// no data: every sphere of a scene is this one geometry, placed by its
// shape's transform (contracts/transform.h), whose translation is the sphere's center
// and whose scale is its radius. Its acceleration structure is one box,
// built once and shared by every sphere (metal/acceleration/
// scene_acceleration.h). The exact hit and the normal are the shader half
// (metal/shapes/sphere.metal.h).

#include "core/contracts/float3.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"

namespace serenity {
namespace shapes {

#if !defined(__METAL_VERSION__)
// Its extent in object space: [-1, 1] on each axis.
inline Bounds sphere_bounds() {
    return Bounds{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};
}

// Whether the sphere `transform` places meets or touches the world box
// `box`, exactly: the point of the box nearest the center (the center
// clamped to it) is within the radius. Computed in double, so a scene's
// numbers, all floats, give an exact answer at a distance of 0. The
// transform is a translation and a uniform scale (contracts/transform.h).
bool sphere_touches(const contracts::Transform& transform, const Bounds& box);
#endif

}  // namespace shapes
}  // namespace serenity
