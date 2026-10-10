#pragma once

// Axis: Acceleration (shader half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any shape blocks a shadow
// ray. Metal's hardware walks the structure of bounding boxes and
// hands back each box the ray enters as a candidate; the shape's own exact
// test (shapes/shapes.metal.h) decides whether, and where, the ray hits
// what is inside, and a hit nearer than the nearest so far is committed.
// Kind-blind: it names no shape kind.
//
// One level (scene_acceleration.h): a candidate is shape i's box, its
// primitive index i. The exact test carries the world's ray into the
// shape's object space itself (shapes/shapes.metal.h), and the t it finds is
// the world's (contracts/transform.h), so it is committed and compared as it
// is. What a hit reports is the shape and t; where on it, in the world, is
// surface_interaction()'s.
//
// A ray is Metal's own bundle, metal::raytracing::ray: origin, unit
// direction, and the open interval (min_distance, max_distance) its hits lie
// in (I.23), which the query takes as it is.

#include <metal_raytracing>
#include <metal_stdlib>

#include "metal/shapes/crossing.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {

// The far end of a ray that runs until it meets something: the largest
// finite float. Not INFINITY: every shader compiles with Metal's default
// fast math, under which an infinite operand makes a comparison's result
// undefined (no-infs-fp-math), and the shapes' exact tests compare their t
// against this bound.
constant constexpr float unbounded = metal::numeric_limits<float>::max();

struct Hit {
    bool found;
    float t;         // the ray's parameter: its distance, the direction being unit
    uint primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
};

// The nearest shape along `r`, its direction unit length.
inline Hit trace(metal::raytracing::primitive_acceleration_structure structure, Shapes shapes,
                 metal::raytracing::ray r) {
    using namespace metal::raytracing;
    intersection_params params;
    params.assume_geometry_type(geometry_type::bounding_box);
    params.accept_any_intersection(false);

    intersection_query<> query;
    query.reset(r, structure, params);
    ray nearer = r;  // hits nearer than the nearest so far
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::bounding_box) {
            continue;
        }
        const Crossing crossing = intersect_shape(shapes, query.get_candidate_primitive_id(), nearer);
        if (crossing.found) {
            query.commit_bounding_box_intersection(crossing.t);
            nearer.max_distance = crossing.t;
        }
    }

    const bool found = query.get_committed_intersection_type() == intersection_type::bounding_box;
    return Hit{found, found ? query.get_committed_distance() : 0.0f,
               found ? query.get_committed_primitive_id() : 0u};
}

// Whether any shape but `ignore` lies along `r`: a shadow ray's question. It
// stops at the first such shape, whichever it is, rather than searching for
// the nearest.
inline bool occluded(metal::raytracing::primitive_acceleration_structure structure, Shapes shapes,
                     metal::raytracing::ray r, uint ignore) {
    using namespace metal::raytracing;
    intersection_params params;
    params.assume_geometry_type(geometry_type::bounding_box);
    params.accept_any_intersection(true);

    intersection_query<> query;
    query.reset(r, structure, params);
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::bounding_box) {
            continue;
        }
        const uint shape = query.get_candidate_primitive_id();
        if (shape != ignore && intersect_shape(shapes, shape, r).found) {
            query.abort();
            return true;
        }
    }
    return false;
}

}  // namespace shaders
}  // namespace serenity
