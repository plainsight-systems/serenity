#pragma once

#include <cstdint>
#include <vector>

#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "core/shapes/sphere.h"
#include "core/shapes/transform.h"

namespace serenity::shapes {

// Axis: Shape.
//
// Every shape in a scene: one record and one transform per shape, in the
// scene's order, and one geometry array per kind that has data (primitive.h).
// The family's one view of all its kinds, so what consumes every shape, the
// acceleration build and the scene reader's checks, names no kind
// (metal/acceleration/scene_acceleration.h, core/scene/scene.h).
//
// The transforms are each shape at rest, as the scene file places it. A
// shape that moves is placed anew each frame, in a copy of them the GPU
// backend keeps per frame (core/scene/animate.h); the records and the
// geometry never change.
//
// CPU only: shaders read the records, the transforms and the arrays, each in
// its shared layout, from the GPU (metal/scene/scene_buffers.h).
struct Shapes {
    std::vector<ShapeRecord> records;    // shape i is records[i]
    std::vector<Transform> transforms;   // and transforms[i]
    std::vector<BoxData> boxes;          // the box geometries
};

// The extent of geometry `record` names, in its object space, by its kind's
// bounds: the one box its acceleration structure holds. The record must
// index its kind's array; the scene reader makes it so. A switch with no
// default, so a kind without bounds fails the build.
Bounds object_bounds(const Shapes& shapes, const ShapeRecord& record);

// Whether shape `shape`, where its transform places it, meets or touches the
// world box `box`, by its kind's exact test (sphere.h, box.h): what the scene
// reader asks of every still shape before it lets a moving one sweep `box`
// (core/scene/scene.h). A switch with no default, as above.
//
// Not performance-sensitive: once per geometry at start-up, and once per
// still shape per moving shape, at load.
bool touches(const Shapes& shapes, std::uint32_t shape, const Bounds& box);

}  // namespace serenity::shapes
