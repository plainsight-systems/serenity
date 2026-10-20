#pragma once

// Axis: Shape (shader half of shapes/shapes.h).
//
// Every shape, as a shader reads it: the records, the frame's transforms,
// and one geometry array per kind that has data (shapes/primitive.h). Given
// a shape, its exact hit, its normal and its material, by its kind: the one
// place a shader dispatches on shape kind, so the tracing loop
// (acceleration/trace.cuh) names none. The switches have no default, so
// a kind added to ShapeKind and not here fails Metal's compile (-Werror), and
// here, where nvcc does not warn of an enumerator a switch leaves out, traps
// (device/layout.cuh, unhandled_kind).
//
// The exact hit is in the shape's object space: intersect_shape() carries
// the world's ray there by the inverse of the shape's transform, its
// direction not renormalized, so its t is the world's
// (contracts/transform.h), and tests it against the kind's geometry about
// the origin. surface_interaction() takes the world's hit point, carries it
// into object space the same way, asks the kind to put it back on its
// surface and for its object-space normal there, places that point in the
// world and turns that normal by the transform's rotation into the world's.
//
// Why the point is put back (decided 2026-10-11, with the ray offset:
// docs/research/2026-10-11-ray-offset.md). The hit the ray gives, origin +
// t d, is off the surface by t's rounding, which grows with t, so a point
// found from 100 m away is off by more than the step a leaving ray is
// offset by (integrator/surface.cuh). Put back on the surface where the
// kind defines it and placed from there, it is off by a few ulps of its own
// coordinates, as a triangle's point interpolated from its vertices is in
// Falcor and RTXDI. A box's lies on its face's plane exactly, its transform
// the identity (shapes/box.h). Per hit: a box's plane chosen by two
// selects, and one transform of a point (three dot products and an add);
// a sphere's normal, already the normalized point, is now also its point.
//
// Translated line for line from Metal's metal/shapes/shapes.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/surface_interaction.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/ray.cuh"
#include "cuda/math/transform.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/shapes/box.cuh"
#include "cuda/shapes/crossing.cuh"
#include "cuda/shapes/sphere.cuh"

namespace serenity {
namespace shaders {

struct Shapes {
  const serenity::shapes::ShapeRecord* records;
  const serenity::contracts::Transform*
      transforms;  // as this frame places them
  const serenity::shapes::BoxData* boxes;
};

// The exact crossing of world ray `world` with shape `shape`'s geometry,
// tested in its object space; its t is the world's.
__device__ inline Crossing intersect_shape(Shapes shapes, uint shape,
                                           Ray world) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const Ray object = transform_ray_to_object(shapes.transforms[shape], world);
  switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
      return intersect_sphere(object);
    case serenity::shapes::ShapeKind::box:
      return intersect_box(shapes.boxes[record.geometry], object);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return no_crossing();
}

// A point found near shape `shape`'s surface, in its own coordinates, put
// back on it, and the outward normal there, both in its own coordinates.
struct OnShape {
  float3 point;
  float3 normal;
};

// By its kind: a sphere's along the same direction from its center, a box's
// onto the plane of the face it is nearest.
__device__ inline OnShape on_shape(Shapes shapes, uint shape, float3 at) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere: {
      const float3 on = sphere_surface_point(at);
      return OnShape{on, on};
    }
    case serenity::shapes::ShapeKind::box: {
      const serenity::shapes::BoxData box = shapes.boxes[record.geometry];
      const float3 normal = box_normal(box, at);
      return OnShape{box_surface_point(box, at, normal), normal};
    }
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return OnShape{at, make_float3(0.0f, 1.0f, 0.0f)};
}

// Contract 1: where a ray along `direction` met shape `shape` near world
// `point`, filled by the shape (contracts/surface_interaction.h). The
// shading normal is the geometric one for every kind so far. The point is
// put back on the shape's surface in its own coordinates and placed in the
// world from there (above): on it to a few ulps of its own coordinates,
// however far the ray came, as leaving it needs (integrator/
// metal/shapes/surface.metal.h). The object-space point is found once, for the
// normal, the world's point and the textures laid on the shape.
__device__ inline serenity::contracts::SurfaceInteraction surface_interaction(
    Shapes shapes, uint shape, float3 point, float3 direction) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const serenity::contracts::Transform placed = shapes.transforms[shape];
  const OnShape on =
      on_shape(shapes, shape, transform_to_object(placed, point));
  const float3 normal = normalize(transform_direction(placed, on.normal));
  serenity::contracts::SurfaceInteraction s{};
  s.position = to_packed(transform_to_world(placed, on.point));
  s.material = record.material;
  s.geometric_normal = to_packed(normal);
  s.primitive = shape;
  s.shading_normal = to_packed(normal);
  s.flags = dot(direction, normal) < 0.0f
                ? serenity::contracts::arrived_from_outside
                : 0u;
  s.object_position = to_packed(on.point);
  s.interior = record.interior;
  return s;
}

}  // namespace shaders
}  // namespace serenity
