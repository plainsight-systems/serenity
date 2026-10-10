#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "core/shapes/sphere.h"

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
// backend keeps per frame (core/animation/animate.h); the records and the
// geometry never change.
//
// CPU only: shaders read the records, the transforms and the arrays, each in
// its shared layout, from the GPU (metal/scene/scene_buffers.h).
struct Shapes {
    std::vector<ShapeRecord> records;    // shape i is records[i]
    std::vector<contracts::Transform> transforms;  // and transforms[i]
    std::vector<BoxData> boxes;          // the box geometries
};

// The extent of geometry `record` names, in its object space, by its kind's
// bounds: the one box its acceleration structure holds. The record must
// index its kind's array; the scene reader makes it so. A switch with no
// default, so a kind without bounds fails the build. The record, 16 bytes,
// is taken by value (F.16). Here and below, a record whose kind is no
// ShapeKind is refused by std::logic_error, never answered with a value
// that looks like a shape's (P.6), and an index past an array by .at().
Bounds object_bounds(const Shapes& shapes, ShapeRecord record);

// The world box that holds a geometry whose object-space bounds are
// `object`, placed by `transform`: the box about the transformed center
// whose half extent on each axis is the absolute linear part times the
// object's half extent, the tight box of a transformed box, rotation
// included (Arvo, "Transforming Axis-Aligned Bounding Boxes", Graphics Gems
// 1990). In double, rounded outward, so the float box never cuts into what
// it holds: a ray that grazes a sphere's silhouette still enters its box.
// What the acceleration structure holds for each shape
// (metal/acceleration/scene_acceleration.h), once for a still shape, each
// frame for a moving one.
//
// Performance-sensitive: once per moving shape per frame. Some 30 flops,
// nothing allocated.
Bounds world_bounds(const Bounds& object, const contracts::Transform& transform);

// The distance from world `point` to shape `shape`'s surface, where its
// transform places it, by its kind's exact test (sphere.h, box.h): 0 on it,
// negative inside. In double. A switch with no default, as below. For one
// shape; the many questions a load asks of every still shape are
// StillShapes' (below).
double distance(const Shapes& shapes, std::uint32_t shape, contracts::Float3 point);

// Whether shape `shape`, where its transform places it, meets or touches the
// world box `box`, by its kind's exact test (sphere.h, box.h): what the scene
// reader asks of every still shape before it lets a moving one sweep `box`
// (core/scene/scene.h). A switch with no default, as above.
//
// Not performance-sensitive: once per geometry at start-up, and once per
// still shape per moving shape, at load.
bool touches(const Shapes& shapes, std::uint32_t shape, const Bounds& box);

// Some of a scene's shapes, placed once, for the questions a load asks of
// them over and over: the distance from a point to the nearest, and whether
// any touches a box. What the scene answers contract 11 with, over its
// still shapes (core/scene/scene.h, contracts/obstacles.h), as each flight
// keeps clear of them: once per still shape per sample of a flight, some
// 10^4 to 10^5 samples a firefly (core/animation/flight.h), the load's
// largest cost.
//
// Optimization (CDSA.32, CACHE.4): each shape's transform is checked for a
// rotation and widened to double once, here, not at every question, and
// each kind is kept in an array of what its test reads, a sphere's center
// and radius, a box's world-space faces, scanned in order with no dispatch
// on the kind and no bounds check per shape. The arithmetic is distance()'s
// and touches()' own (one function per kind serves both), so the answers are
// theirs to the bit: the nearest of the same numbers, and whether any is
// within. Measured in docs/research/2026-10-10-flight-load.md.
//
// Read-only once made, so asked safely from many threads at once
// (contract 11). Throws std::invalid_argument, as distance() would, for a
// transform with a rotation, and std::logic_error for a kind no
// enumerator names.
class StillShapes {
public:
    StillShapes(const Shapes& shapes, std::span<const std::uint32_t> which);

    // The distance from `point` to the nearest shape's surface; infinity
    // if there are none.
    double distance(contracts::Float3 point) const noexcept;

    // Whether any shape meets or touches `box`.
    bool touches(const Bounds& box) const noexcept;

    // A sphere where its transform places it, and a box's world-space
    // faces, in double: what each kind's tests read.
    struct PlacedSphere {
        std::array<double, 3> center{};
        double radius = 0.0;
    };
    struct PlacedBox {
        std::array<double, 3> low{};
        std::array<double, 3> high{};
    };

private:
    std::vector<PlacedSphere> spheres_;
    std::vector<PlacedBox> boxes_;
};

}  // namespace serenity::shapes
