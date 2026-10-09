#pragma once

// Axis: Shape (shader half of shapes/shapes.h).
//
// Every shape, as a shader reads it: the records and one array per kind
// (shapes/primitive.h). Given a primitive, its exact hit, its normal and its
// material, by its kind: the one place a shader dispatches on shape kind, so
// the tracing loop (acceleration/trace.metal.h) names none. The switches
// have no default, so a kind added to ShapeKind and not here fails to
// compile (-Werror).

#include <metal_stdlib>

#include "core/contracts/surface_interaction.h"
#include "core/shapes/primitive.h"
#include "metal/shapes/box.metal.h"
#include "metal/shapes/sphere.metal.h"

namespace serenity {
namespace shaders {

struct Shapes {
    device const serenity::shapes::PrimitiveRecord* records;
    device const serenity::shapes::SphereData* spheres;
    device const serenity::shapes::BoxData* boxes;
};

inline bool intersect_shape(Shapes shapes, uint primitive, float3 origin, float3 direction, float t_min, float t_max,
                            thread float& t) {
    const serenity::shapes::PrimitiveRecord record = shapes.records[primitive];
    switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
        return intersect_sphere(shapes.spheres[record.index], origin, direction, t_min, t_max, t);
    case serenity::shapes::ShapeKind::box:
        return intersect_box(shapes.boxes[record.index], origin, direction, t_min, t_max, t);
    }
    return false;
}

inline float3 shape_normal(Shapes shapes, uint primitive, float3 point) {
    const serenity::shapes::PrimitiveRecord record = shapes.records[primitive];
    switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
        return sphere_normal(shapes.spheres[record.index], point);
    case serenity::shapes::ShapeKind::box:
        return box_normal(shapes.boxes[record.index], point);
    }
    return float3(0.0f, 1.0f, 0.0f);
}

inline uint shape_material(Shapes shapes, uint primitive) {
    const serenity::shapes::PrimitiveRecord record = shapes.records[primitive];
    switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
        return shapes.spheres[record.index].material;
    case serenity::shapes::ShapeKind::box:
        return shapes.boxes[record.index].material;
    }
    return 0;
}

// Contract 1: where a ray along `direction` met primitive `primitive` at
// `point`, filled by the shape (contracts/surface_interaction.h). The
// shading normal is the geometric one for every kind so far.
inline serenity::contracts::SurfaceInteraction surface_interaction(Shapes shapes, uint primitive, float3 point,
                                                                    float3 direction) {
    const float3 normal = shape_normal(shapes, primitive, point);
    serenity::contracts::SurfaceInteraction s;
    s.position = to_packed(point);
    s.material = shape_material(shapes, primitive);
    s.geometric_normal = to_packed(normal);
    s.primitive = primitive;
    s.shading_normal = to_packed(normal);
    s.flags = metal::dot(direction, normal) < 0.0f ? serenity::contracts::arrived_from_outside : 0u;
    return s;
}

}  // namespace shaders
}  // namespace serenity
