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
// into object space the same way, asks the kind to put it back on its
// surface and for its object-space normal there, places that point in the
// world and turns that normal by the transform's rotation into the world's.
//
// Why the point is put back (decided 2026-10-11, with the ray offset:
// docs/research/2026-10-11-ray-offset.md). The hit the ray gives, origin +
// t d, is off the surface by t's rounding, which grows with t, so a point
// found from 100 m away is off by more than the step a leaving ray is
// offset by (integrator/surface.metal.h). Put back on the surface where the
// kind defines it and placed from there, it is off by a few ulps of its own
// coordinates, as a triangle's point interpolated from its vertices is in
// Falcor and RTXDI. A box's lies on its face's plane exactly, its transform
// the identity (shapes/box.h). Per hit: a box's plane chosen by two
// selects, and one transform of a point (three dot products and an add);
// a sphere's normal, already the normalized point, is now also its point.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/surface_interaction.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"
#include "metal/device/layout.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/shapes/box.metal.h"
#include "metal/shapes/crossing.metal.h"
#include "metal/shapes/sphere.metal.h"

namespace serenity {
namespace shaders {

struct Shapes {
  constant serenity::shapes::ShapeRecord* records;
  constant serenity::contracts::Transform*
      transforms;  // as this frame places them
  constant serenity::shapes::BoxData* boxes;
};

// The exact crossing of world ray `world` with shape `shape`'s geometry,
// tested in its object space; its t is the world's.
inline Crossing intersect_shape(Shapes shapes, uint shape,
                                metal::raytracing::ray world) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const metal::raytracing::ray object =
      transform_ray_to_object(shapes.transforms[shape], world);
  switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
      return intersect_sphere(object);
    case serenity::shapes::ShapeKind::box:
      return intersect_box(shapes.boxes[record.geometry], object);
  }
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
inline OnShape on_shape(Shapes shapes, uint shape, float3 at) {
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
  return OnShape{at, float3(0.0f, 1.0f, 0.0f)};
}

// Contract 1: where a ray along `direction` met shape `shape` near world
// `point`, filled by the shape (contracts/surface_interaction.h). The
// shading normal is the geometric one for every kind so far. The point is
// put back on the shape's surface in its own coordinates and placed in the
// world from there (above): on it to a few ulps of its own coordinates,
// however far the ray came, as leaving it needs (integrator/
// surface.metal.h). The object-space point is found once, for the normal,
// the world's point and the textures laid on the shape.
inline serenity::contracts::SurfaceInteraction surface_interaction(
    Shapes shapes, uint shape, float3 point, float3 direction) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const serenity::contracts::Transform placed = shapes.transforms[shape];
  const OnShape on =
      on_shape(shapes, shape, transform_to_object(placed, point));
  const float3 normal =
      metal::normalize(transform_direction(placed, on.normal));
  serenity::contracts::SurfaceInteraction s{};
  s.position = to_packed(transform_to_world(placed, on.point));
  s.material = record.material;
  s.geometric_normal = to_packed(normal);
  s.primitive = shape;
  s.shading_normal = to_packed(normal);
  s.flags = metal::dot(direction, normal) < 0.0f
                ? serenity::contracts::arrived_from_outside
                : 0u;
  s.object_position = to_packed(on.point);
  s.interior = record.interior;
  return s;
}

// Where a traced ray met a shape's mesh (acceleration/trace.metal.h): the
// point on the triangle it hit, in the world, from the triangle's vertices
// and the hit's barycentrics, as Falcor and RTXDI find a hit's point; and
// the triangle's outward unit normal, in the world. The CUDA backend's
// (cuda/shapes/shapes.cuh).
struct MeshPoint {
  float3 point;
  float3 face;
};

// Shape `shape`'s instance transform: its transform after its mesh's own
// placement, the identity for a sphere and the unit box onto the box's
// corners for a box (core/shapes/shape_meshes.h, instance_transform, which
// the host gives the instance itself).
inline serenity::contracts::Transform instance_placement(Shapes shapes,
                                                         uint shape) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  serenity::contracts::Transform placed = shapes.transforms[shape];
  if (record.kind != serenity::shapes::ShapeKind::box) {
    return placed;
  }
  const serenity::shapes::BoxData box = shapes.boxes[record.geometry];
  const float3 low = float3(box.min.x, box.min.y, box.min.z);
  const float3 extent = float3(box.max.x, box.max.y, box.max.z) - low;
  serenity::contracts::Transform out = placed;
  for (uint r = 0u; r < 3u; ++r) {
    out.m[r][3] = placed.m[r][3] + placed.m[r][0] * low.x +
                  placed.m[r][1] * low.y + placed.m[r][2] * low.z;
    for (uint k = 0u; k < 3u; ++k) {
      out.m[r][k] = placed.m[r][k] * extent[k];
    }
  }
  return out;
}

// Contract 1 for a traced hit on shape `shape`'s mesh, the ray along
// `direction`, as the CUDA backend makes it (cuda/shapes/shapes.cuh;
// docs/research/2026-10-20-nvidia-hardware-design.md, "The spheres as
// triangles"). The point is the triangle's: on the surface the ray traced
// to a few ulps of its own coordinates, which is what a leaving ray's
// offset assumes (integrator/surface.metal.h), wherever the shape is and
// whichever way the ray leaves. The geometric normal is the triangle's, so
// a ray leaves on the side of the surface it traced and arrives from the
// side it crossed. The shading normal is the shape's own at the point's
// place on it, as a smooth mesh's interpolated normal stands for it: a
// sphere's along its radius, a box's its face's (the triangle's). So the
// BSDF sees the sphere and the ray sees the mesh, which differ by under a
// degree at the levels used (core/shapes/mesh_levels.h). The object-space
// position is the shape's own point too, so a texture laid on it (contract
// 1's object_position) is continuous across the triangles.
inline serenity::contracts::SurfaceInteraction surface_interaction(
    Shapes shapes, uint shape, MeshPoint at, float3 direction) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const serenity::contracts::Transform placed = shapes.transforms[shape];
  const OnShape on =
      on_shape(shapes, shape, transform_to_object(placed, at.point));
  serenity::contracts::SurfaceInteraction s{};
  s.position = to_packed(at.point);
  s.material = record.material;
  s.geometric_normal = to_packed(at.face);
  s.primitive = shape;
  s.shading_normal =
      to_packed(metal::normalize(transform_direction(placed, on.normal)));
  s.flags = metal::dot(direction, at.face) < 0.0f
                ? serenity::contracts::arrived_from_outside
                : 0u;
  s.object_position = to_packed(on.point);
  s.interior = record.interior;
  return s;
}

}  // namespace shaders
}  // namespace serenity
