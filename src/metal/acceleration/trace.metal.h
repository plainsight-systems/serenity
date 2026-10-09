#pragma once

// Axis: Acceleration (shader half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any shape blocks a shadow
// ray. Metal's hardware walks the two-level structure of bounding boxes and
// hands back each box the ray enters as a candidate; the shape's own exact
// test (shapes/shapes.metal.h) decides whether, and where, the ray hits
// what is inside, and a hit nearer than the nearest so far is committed.
// Kind-blind: it names no shape kind.
//
// The structure is two levels (scene_acceleration.h): a candidate is a box
// of some geometry, reached through an instance, whose user ID is its shape
// (shapes/primitive.h). The exact test takes the ray the query hands over
// for that candidate, in the instance's object space
// (get_candidate_ray_origin and _direction), and the t it finds is the
// world's (shapes/transform.h), so it is committed and compared as it is.
// What a hit reports is the shape and t; where on it, in the world, is
// surface_interaction()'s (shapes/shapes.metal.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {

struct Hit {
    bool found;
    float t;         // distance along the ray
    uint primitive;  // the shape, as primitive i is shape i
};

// `direction` is unit length; hits are in (t_min, t_max).
inline Hit trace(metal::raytracing::primitive_acceleration_structure structure, Shapes shapes, float3 origin,
                 float3 direction, float t_min, float t_max) {
    using namespace metal::raytracing;
    intersection_params params;
    params.assume_geometry_type(geometry_type::bounding_box);
    params.accept_any_intersection(false);

    intersection_query<> query;
    query.reset(ray(origin, direction, t_min, t_max), structure, params);
    float nearest = t_max;
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::bounding_box) {
            continue;
        }
        const uint primitive = query.get_candidate_primitive_id();
        float t = 0.0f;
        if (intersect_shape(shapes, primitive, origin, direction, t_min, nearest, t)) {
            query.commit_bounding_box_intersection(t);
            nearest = t;
        }
    }

    Hit hit;
    hit.found = query.get_committed_intersection_type() == intersection_type::bounding_box;
    hit.t = hit.found ? query.get_committed_distance() : 0.0f;
    hit.primitive = hit.found ? query.get_committed_primitive_id() : 0u;
    return hit;
}

// Whether anything but primitive `ignore` lies along the ray within
// (t_min, t_max): a shadow ray's question. It stops at the first such shape,
// whichever it is, rather than searching for the nearest.
inline bool occluded(metal::raytracing::primitive_acceleration_structure structure, Shapes shapes, float3 origin,
                     float3 direction, float t_min, float t_max, uint ignore) {
    using namespace metal::raytracing;
    intersection_params params;
    params.assume_geometry_type(geometry_type::bounding_box);
    params.accept_any_intersection(true);

    intersection_query<> query;
    query.reset(ray(origin, direction, t_min, t_max), structure, params);
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::bounding_box) {
            continue;
        }
        const uint primitive = query.get_candidate_primitive_id();
        float t = 0.0f;
        if (primitive != ignore && intersect_shape(shapes, primitive, origin, direction, t_min, t_max, t)) {
            query.abort();
            return true;
        }
    }
    return false;
}

}  // namespace shaders
}  // namespace serenity
