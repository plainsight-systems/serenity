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

}  // namespace shaders
}  // namespace serenity
