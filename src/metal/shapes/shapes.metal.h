#pragma once

// Axis: Shape (shader half of shapes/shapes.h).
//
// Every shape, as a shader reads it: the records, the frame's transforms,
// and one geometry array per kind that has data (shapes/primitive.h). Given
// a shape, its exact hit, its normal and its material, by its kind: the one
// place a shader dispatches on shape kind, so the tracing loop
// (acceleration/trace.metal.h) names none. The switches have no default, so
// a kind added to ShapeKind and not here fails to compile (-Werror).
//
// The exact hit is in the shape's object space: intersect_shape() carries
// the world's ray there by the inverse of the shape's transform, its
// direction not renormalized, so its t is the world's
// (contracts/transform.h), and tests it against the kind's geometry about
// the origin. surface_interaction() takes the world's hit point, carries it
// into object space the same way, asks the kind for its object-space normal
// there, and turns that normal by the transform's rotation into the world's.

#include <metal_stdlib>

#include "core/contracts/surface_interaction.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"
#include "metal/device/layout.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/shapes/box.metal.h"
#include "metal/shapes/sphere.metal.h"

namespace serenity {
namespace shaders {

struct Shapes {
    device const serenity::shapes::ShapeRecord* records;
    device const serenity::contracts::Transform* transforms;  // as this frame places them
    device const serenity::shapes::BoxData* boxes;
};

// The exact hit of a world ray on shape `shape`'s geometry, tested in its
// object space; `t` is the world's.
inline bool intersect_shape(Shapes shapes, uint shape, float3 origin, float3 direction, float t_min, float t_max,
                            thread float& t) {
    const serenity::shapes::ShapeRecord record = shapes.records[shape];
    float3 o;
    float3 d;
    transform_ray_to_object(shapes.transforms[shape], origin, direction, o, d);
    switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
        return intersect_sphere(o, d, t_min, t_max, t);
    case serenity::shapes::ShapeKind::box:
        return intersect_box(shapes.boxes[record.geometry], o, d, t_min, t_max, t);
    }
    return false;
}

// The outward normal of shape `shape` at world `point`, in the world: found
// in object space, turned by the transform's rotation.
inline float3 shape_normal(Shapes shapes, uint shape, float3 point) {
    const serenity::shapes::ShapeRecord record = shapes.records[shape];
    const serenity::contracts::Transform placed = shapes.transforms[shape];
    const float3 at = transform_to_object(placed, point);
    float3 normal = float3(0.0f, 1.0f, 0.0f);
    switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
        normal = sphere_normal(at);
        break;
    case serenity::shapes::ShapeKind::box:
        normal = box_normal(shapes.boxes[record.geometry], at);
        break;
    }
    return metal::normalize(transform_direction(placed, normal));
}

// Contract 1: where a ray along `direction` met shape `shape` at world
// `point`, filled by the shape (contracts/surface_interaction.h). The
// shading normal is the geometric one for every kind so far.
inline serenity::contracts::SurfaceInteraction surface_interaction(Shapes shapes, uint shape, float3 point,
                                                                    float3 direction) {
    const float3 normal = shape_normal(shapes, shape, point);
    serenity::contracts::SurfaceInteraction s;
    s.position = to_packed(point);
    s.material = shapes.records[shape].material;
    s.geometric_normal = to_packed(normal);
    s.primitive = shape;
    s.shading_normal = to_packed(normal);
    s.flags = metal::dot(direction, normal) < 0.0f ? serenity::contracts::arrived_from_outside : 0u;
    return s;
}

}  // namespace shaders
}  // namespace serenity
