#pragma once

#include <vector>

#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "core/shapes/sphere.h"

namespace serenity::shapes {

// Axis: Shape.
//
// Every shape in a scene: one record per shape, in the scene's order, and
// one array per kind that the records index (primitive.h). The family's one
// view of all its kinds, so what consumes every shape, the acceleration
// build, takes their bounds from here and names no kind
// (metal/acceleration/primitives.h).
//
// CPU only: shaders read the records and the arrays, each in its shared
// layout, from the GPU (metal/scene/scene_buffers.h).
struct Shapes {
    std::vector<PrimitiveRecord> records;  // primitive i is records[i]
    std::vector<SphereData> spheres;
    std::vector<BoxData> boxes;
};

// The bounds of every shape, in primitive order: element i bounds the shape
// records[i] names, by its kind's bounds(). Each record must index its
// kind's array; the scene reader makes them so. The mapping from kind to
// bounds is a switch with no default, so a kind without bounds fails the
// build.
//
// Not performance-sensitive: once per build of the acceleration structure.
std::vector<Bounds> bounds(const Shapes& shapes);

}  // namespace serenity::shapes
