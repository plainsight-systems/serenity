#pragma once

// Axis: Shape (a point on a shape, in the shape's own coordinates).
//
// A point stored by the shape it is on rather than by where it is, so that
// where it is can be found again against where the shape is now: ReSTIR
// PT's reconnection vertex (contract 15, core/contracts/path_reservoir.h,
// `point`), as contract 4 stores a light's point (light_sample.h).
//
//   sphere  the point's direction from the center in the unit sphere's own
//           space, by the equal-area octahedral map in two 16-bit numbers
//           (math/warp.metal.h, encode_unit): within 10^-4 radians, a
//           micrometer on a marble.
//   box     the face, 0 to 5, the axis times 2 plus 1 on the maximum's
//           side; and the point's two other coordinates in the box's own
//           space as 16-bit fractions of its extent, the lower axis in the
//           low half: within 1 / 131070 of the extent, 8 um across a 1 m
//           table.
//
// Decoding puts the point on the shape's surface in its own space, then
// places it with the shape's transform: the caller asks for the surface
// there by surface_interaction() (shapes.metal.h), which takes it as a hit
// it reprojects. Pure (F.8); a switch on the kind with no default, as
// shapes.metal.h's, so a kind added and not stored here fails to compile.

#include <metal_stdlib>

#include "core/contracts/path_reservoir.h"
#include "core/shapes/primitive.h"
#include "metal/device/layout.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/shapes/box.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/shapes/sphere.metal.h"

namespace serenity {
namespace shaders {

// a 16-bit fraction's largest code
constant constexpr float shape_point_scale = 65535.0f;

// `world`, a point on shape `shape`, in that shape's coordinates.
inline serenity::contracts::ShapePoint encode_shape_point(Shapes shapes,
                                                          uint shape,
                                                          float3 world) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const float3 object = transform_to_object(shapes.transforms[shape], world);
  switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
      return serenity::contracts::ShapePoint{
          encode_unit(sphere_surface_point(object)), 0u};
    case serenity::shapes::ShapeKind::box: {
      const serenity::shapes::BoxData box = shapes.boxes[record.geometry];
      const float3 normal = box_normal(box, object);
      const uint axis = normal.x != 0.0f ? 0u : (normal.y != 0.0f ? 1u : 2u);
      const float3 low = to_float3(box.min);
      const float3 across =
          metal::saturate((object - low) / (to_float3(box.max) - low));
      const float2 uv =
          axis == 0u ? across.yz : (axis == 1u ? across.xz : across.xy);
      const uint2 codes = uint2(metal::rint(uv * shape_point_scale));
      return serenity::contracts::ShapePoint{
          codes.x | (codes.y << 16u),
          axis * 2u + (normal[axis] > 0.0f ? 1u : 0u)};
    }
  }
  return serenity::contracts::ShapePoint{0u, 0u};
}

// The world point `point` stands for on shape `shape`, as the shape is placed
// now.
inline float3 decode_shape_point(Shapes shapes, uint shape,
                                 serenity::contracts::ShapePoint point) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  float3 object = float3(0.0f);
  switch (record.kind) {
    case serenity::shapes::ShapeKind::sphere:
      object = decode_unit(point.point);
      break;
    case serenity::shapes::ShapeKind::box: {
      const serenity::shapes::BoxData box = shapes.boxes[record.geometry];
      const float3 low = to_float3(box.min);
      const float3 high = to_float3(box.max);
      const float2 uv =
          float2(float(point.point & 0xffffu), float(point.point >> 16u)) /
          shape_point_scale;
      const uint axis = metal::min(point.face / 2u, 2u);
      const float side = (point.face & 1u) != 0u ? high[axis] : low[axis];
      const float3 at = metal::mix(
          low, high,
          axis == 0u
              ? float3(0.0f, uv)
              : (axis == 1u ? float3(uv.x, 0.0f, uv.y) : float3(uv, 0.0f)));
      object = axis == 0u   ? float3(side, at.y, at.z)
               : axis == 1u ? float3(at.x, side, at.z)
                            : float3(at.x, at.y, side);
      break;
    }
  }
  return transform_to_world(shapes.transforms[shape], object);
}

}  // namespace shaders
}  // namespace serenity
