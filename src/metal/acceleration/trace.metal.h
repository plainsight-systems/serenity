#pragma once

// Axis: Acceleration (shader half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any occluder blocks a
// shadow ray (contract 19, core/contracts/ray_mask.h), as the M3's
// ray-tracing unit answers them (docs/research/2026-10-23-apple-silicon-
// design.md, section 6, change 1): Metal's intersector over the instance
// structure of the shapes' triangle meshes, every geometry opaque, so the
// hardware walks the structure, carries the ray into each instance's space
// and tests its triangles itself, running no intersection function, as
// Apple asks of the M3 ("use the intersector object API whenever
// possible"; "maximize opaque triangle tests") and Blender Cycles traces
// there. The CUDA backend's trace() and occluded() (cuda/acceleration/
// trace.cuh), with the same signatures. Kind-blind: it names no shape
// kind.
//
// A hit names its shape by its instance's user id, the shape's index, and
// gives the triangle's point and outward normal (mesh_point()): its
// vertices from the hit's per-primitive data (structure_block.h,
// MeshTriangle) and their barycentric weights, placed in the world by the
// shape's transform after its mesh's own placement, which
// surface_interaction() makes contract 1 of (shapes/shapes.metal.h). Its t
// is the mesh's: within the mesh's gap of the sphere's own
// (core/shapes/mesh_levels.h).
//
// The mask is an argument, not a bound value: every site passes a constant
// (I.4: a RayMask, no integer), so the compiler picks the instance bit
// where it is traced, and the threads of a SIMD group at one site trace
// the same one (GPU.4). A shadow ray carries the occluders, which hold no
// light (contract 19), so the light it aims at is never among what it can
// meet: occluded() takes no shape to ignore, and no mask.
//
// Manifold next event estimation's seed is a walk of trace()'s nearest
// hits among the occluders (integrator/mnee.metal.h, seed_segment), so
// every ray in the frame is the intersector's on opaque triangles: no
// intersection function, and no intersection query, which would turn off
// the reorder stage for its whole kernel (Apple, tech talk 111375).
//
// A ray is Metal's own bundle, metal::raytracing::ray: origin, unit
// direction, and the open interval (min_distance, max_distance) its hits lie
// in (I.23), which the intersector takes as it is.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/ray_mask.h"
#include "metal/acceleration/structure_block.h"
#include "metal/math/transform.metal.h"
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

// The structures a frame traces, as a pass binds them (structure_block.h).
using SceneStructure = constant serenity::gpu::StructureBlock*;

// The intersector every trace runs: triangles, opaque, one instance level,
// the hit's triangle data (barycentrics) in its result.
using MeshIntersector =
    metal::raytracing::intersector<metal::raytracing::triangle_data,
                                   metal::raytracing::instancing>;

// The instance bit a mask's rays carry (structure_block.h).
inline uint mask_bit(serenity::contracts::RayMask mask) {
  return mask == serenity::contracts::RayMask::everything
             ? serenity::gpu::everything_bit
             : serenity::gpu::occluder_bit;
}

struct Hit {
  bool found;
  float t;  // the ray's parameter: its distance, the direction being unit
  uint primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
  MeshPoint at;    // where on its mesh (shapes/shapes.metal.h)
  uint part;       // the triangle it met, its index in its mesh
};

// A hit's triangle, its point and outward normal, in the world: its
// vertices in its mesh's space and their barycentric weights (Metal's:
// 1 - u - v, u, v for the triangle's three), placed by the shape's
// instance transform (the shape's transform after its mesh's placement:
// core/shapes/shape_meshes.h, instance_transform), which carries a box's
// unit mesh onto its corners too, so the normal is the placed edges' cross
// product, for any affine placement.
inline MeshPoint mesh_point(Shapes shapes, uint shape,
                            const device serenity::gpu::MeshTriangle& tri,
                            float2 b) {
  const float3 v0 = float3(tri.v[0][0], tri.v[0][1], tri.v[0][2]);
  const float3 v1 = float3(tri.v[1][0], tri.v[1][1], tri.v[1][2]);
  const float3 v2 = float3(tri.v[2][0], tri.v[2][1], tri.v[2][2]);
  const serenity::contracts::Transform placed =
      instance_placement(shapes, shape);
  const float3 object = v0 + b.x * (v1 - v0) + b.y * (v2 - v0);
  const float3 w0 = transform_to_world(placed, v0);
  const float3 w1 = transform_to_world(placed, v1);
  const float3 w2 = transform_to_world(placed, v2);
  return MeshPoint{transform_to_world(placed, object),
                   metal::normalize(metal::cross(w1 - w0, w2 - w0))};
}

// The nearest shape `mask` finds along `r`, its direction unit length.
inline Hit trace(SceneStructure structures, Shapes shapes,
                 metal::raytracing::ray r, serenity::contracts::RayMask mask) {
  using namespace metal::raytracing;
  if (mask == serenity::contracts::RayMask::occluders &&
      structures->occluder_count == 0u) {
    return Hit{false, 0.0f, 0u, MeshPoint{}, 0u};
  }
  MeshIntersector intersector;
  intersector.assume_geometry_type(geometry_type::triangle);
  intersector.force_opacity(forced_opacity::opaque);
  intersector.accept_any_intersection(false);
  const auto result =
      intersector.intersect(r, structures->scene, mask_bit(mask));
  if (result.type != intersection_type::triangle) {
    return Hit{false, 0.0f, 0u, MeshPoint{}, 0u};
  }
  const uint shape = result.user_instance_id;
  const device serenity::gpu::MeshTriangle& tri =
      *static_cast<const device serenity::gpu::MeshTriangle*>(
          result.primitive_data);
  return Hit{true, result.distance, shape,
             mesh_point(shapes, shape, tri, result.triangle_barycentric_coord),
             result.primitive_id};
}

// Whether any occluder lies along `r`: a shadow ray's question. It stops at
// the first, whichever it is, rather than searching for the nearest.
inline bool occluded(SceneStructure structures, Shapes /*shapes*/,
                     metal::raytracing::ray r) {
  using namespace metal::raytracing;
  if (structures->occluder_count == 0u) {
    return false;
  }
  MeshIntersector intersector;
  intersector.assume_geometry_type(geometry_type::triangle);
  intersector.force_opacity(forced_opacity::opaque);
  intersector.accept_any_intersection(true);
  return intersector
             .intersect(r, structures->scene, serenity::gpu::occluder_bit)
             .type == intersection_type::triangle;
}

}  // namespace shaders
}  // namespace serenity
