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
// A hit gives its shape and the triangle's point and outward normal, which
// surface_interaction() makes contract 1 of (shapes/shapes.metal.h), from
// the hit's per-primitive data (structure_block.h, MeshTriangle) and its
// barycentric weights: on a still structure (its instance's user id
// still_instance), the triangle's shape and its vertices in the world
// (triangle_point()); on a moving shape's instance, the instance's user id,
// the shape's index, and its vertices placed in the world by the shape's
// transform after its mesh's own placement (mesh_point()). Its t is the
// mesh's: within the mesh's gap of the sphere's own
// (core/shapes/mesh_levels.h).
//
// The mask is an argument, not a bound value: every site passes a constant
// (I.4: a RayMask, no integer), so the compiler picks the instance bit
// where it is traced, and the threads of a SIMD group at one site trace
// the same one (GPU.4). A shadow ray carries the occluders, which hold no
// light (contract 19), so the light it aims at is never among what it can
// meet: occluded() takes no shape to ignore, and no mask.
//
// Every ray but manifold next event estimation's seed is the intersector's
// on opaque triangles, with no intersection function; the seed traverses
// the occluders' boxes with one of its own (integrator/mnee.metal.h,
// seed_segment). No ray runs the intersection query, which would turn off
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
  uint part;       // the triangle it met, its index in its structure
};

// A triangle's point at barycentric weights `b` (Metal's: 1 - u - v, u, v
// for its three vertices) and its outward normal, the edges' cross product,
// in the space its vertices are in.
inline MeshPoint triangle_point(float3 v0, float3 v1, float3 v2, float2 b) {
  return MeshPoint{v0 + b.x * (v1 - v0) + b.y * (v2 - v0),
                   metal::normalize(metal::cross(v1 - v0, v2 - v0))};
}

inline float3 corner(const device serenity::gpu::MeshTriangle& tri, uint i) {
  return float3(tri.v[i][0], tri.v[i][1], tri.v[i][2]);
}

// A moving shape's hit triangle, its point and outward normal, in the
// world: its vertices in its mesh's space placed by the shape's instance
// transform (the shape's transform after its mesh's placement:
// core/shapes/shape_meshes.h, instance_transform), which carries a box's
// unit mesh onto its corners too, so the normal is the placed edges' cross
// product, for any affine placement.
inline MeshPoint mesh_point(Shapes shapes, uint shape,
                            const device serenity::gpu::MeshTriangle& tri,
                            float2 b) {
  const serenity::contracts::Transform placed =
      instance_placement(shapes, shape);
  return triangle_point(transform_to_world(placed, corner(tri, 0u)),
                        transform_to_world(placed, corner(tri, 1u)),
                        transform_to_world(placed, corner(tri, 2u)), b);
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
  const device serenity::gpu::MeshTriangle& tri =
      *static_cast<const device serenity::gpu::MeshTriangle*>(
          result.primitive_data);
  const float2 b = result.triangle_barycentric_coord;
  if (result.user_instance_id == serenity::gpu::still_instance) {
    return Hit{
        true, result.distance, tri.shape,
        triangle_point(corner(tri, 0u), corner(tri, 1u), corner(tri, 2u), b),
        result.primitive_id};
  }
  const uint shape = result.user_instance_id;
  return Hit{true, result.distance, shape, mesh_point(shapes, shape, tri, b),
             result.primitive_id};
}

// What a test of a segment's nearest shape needs of its hit and no more:
// its distance and its shape (MNEE's chain's visibility, integrator/
// mnee.metal.h, reaches()), with no point or normal built, Apple's least work a
// hit (tech talk 111375).
struct Met {
  bool found;
  float t;
  uint shape;
};

// The nearest occluder along `r` (contract 19), as a walk needs it.
inline Met nearest_occluder(SceneStructure structures,
                            metal::raytracing::ray r) {
  using namespace metal::raytracing;
  if (structures->occluder_count == 0u) {
    return Met{false, 0.0f, 0u};
  }
  MeshIntersector intersector;
  intersector.assume_geometry_type(geometry_type::triangle);
  intersector.force_opacity(forced_opacity::opaque);
  intersector.accept_any_intersection(false);
  const auto result =
      intersector.intersect(r, structures->scene, serenity::gpu::occluder_bit);
  if (result.type != intersection_type::triangle) {
    return Met{false, 0.0f, 0u};
  }
  const uint shape =
      result.user_instance_id == serenity::gpu::still_instance
          ? static_cast<const device serenity::gpu::MeshTriangle*>(
                result.primitive_data)
                ->shape
          : result.user_instance_id;
  return Met{true, result.distance, shape};
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
