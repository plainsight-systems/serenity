#pragma once

// Axis: Acceleration (device half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any shape blocks a shadow
// ray, among the shapes the ray's mask finds (contract 19,
// core/contracts/ray_mask.h): Metal's trace() and occluded()
// (metal/acceleration/trace.metal.h), with the same signatures, as OptiX
// answers them on the RTX 3090 Ti's RT cores
// (docs/research/2026-10-20-nvidia-hardware-design.md, Phase 1).
// Kind-blind: it names no shape kind.
//
// The structure is one instance level over the shapes' triangle meshes
// (scene_acceleration.h): the RT cores walk it, carry the ray into each
// instance's space and test its triangles, and optixTraverse() returns the
// nearest hit in a hit object, which trace() reads, running no program
// at all. The hit names its shape by its instance's id, the shape's
// index, and gives the triangle's point and normal (hit_mesh_point()),
// which surface_interaction() makes contract 1 of (shapes/shapes.cuh). Its
// t is the mesh's: within the mesh's gap of the sphere's own
// (mesh_levels.h).
//
// The mask is an argument, not a bound value: every site passes a constant
// (I.4: a RayMask, no integer), so the compiler picks the instance bit
// where it is traced, and the threads of a warp at one site trace the same
// one (GPU.4).
//
// A shadow ray carries the occluders, which hold no light, so the light
// it aims at is never among the shapes it can meet; the shape occluded()
// is told to ignore is that light, and a triangle has no program to skip
// one by name. A ray carrying everything that names a shape to ignore is
// one this structure cannot answer, and is refused where it runs, by a
// trap, never answered as if nothing were ignored. Any-hit is off on the
// geometry and the ray.

#include <optix.h>

#include "core/contracts/ray_mask.h"
#include "cuda/acceleration/structure_block.h"
#include "cuda/math/ray.cuh"
#include "cuda/math/transform.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/shapes/crossing.cuh"
#include "cuda/shapes/shapes.cuh"

namespace serenity {
namespace shaders {

// The far end of a ray that runs until it meets something: the largest
// finite float, Metal's (metal/acceleration/trace.metal.h), so a ray's
// interval is the same number on both backends.
inline constexpr float unbounded = ::cuda::std::numeric_limits<float>::max();

// The structures a frame traces, as a pass's launch parameters hold them.
using SceneStructure = const serenity::cuda::StructureBlock*;

// No shape ignored (payload 0).
inline constexpr uint ignores_none = 0xffffffffu;

// What a trace carrying one mask traverses: the instance bit it carries;
// `empty`, the mask finds no shape in this scene (occluders, in a scene of
// lights alone), and no trace is made.
struct MaskedStructure {
  OptixTraversableHandle handle;
  uint bit;
  bool empty;
};

__device__ inline MaskedStructure masked(SceneStructure structures,
                                         serenity::contracts::RayMask mask) {
  if (mask == serenity::contracts::RayMask::everything) {
    return MaskedStructure{structures->scene, serenity::cuda::everything_bit,
                           false};
  }
  return MaskedStructure{structures->scene, serenity::cuda::occluder_bit,
                         structures->occluder_count == 0u};
}

struct Hit {
  bool found;
  float t;  // the ray's parameter: its distance, the direction being unit
  uint primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
  MeshPoint at;    // where on its mesh (shapes/shapes.cuh)
};

// The hit object's triangle, its point and outward normal, in the world:
// its vertices in its instance's space and their barycentric weights
// (OptiX's: 1 - b1 - b2, b1, b2 for the triplet's three), placed by the
// instance's transform, which carries a box's unit mesh onto its corners
// too, so the normal is the placed edges' cross product, for any affine
// placement.
__device__ inline MeshPoint hit_mesh_point() {
  float3 v[3];
  optixHitObjectGetTriangleVertexData(v);
  const float2 b = optixHitObjectGetTriangleBarycentrics();
  serenity::contracts::Transform placed{};
  optixHitObjectGetObjectToWorldTransformMatrix(&placed.m[0][0]);
  const float3 object = v[0] + b.x * (v[1] - v[0]) + b.y * (v[2] - v[0]);
  const float3 w0 = transform_to_world(placed, v[0]);
  const float3 w1 = transform_to_world(placed, v[1]);
  const float3 w2 = transform_to_world(placed, v[2]);
  return MeshPoint{transform_to_world(placed, object),
                   normalize(cross(w1 - w0, w2 - w0))};
}

// The nearest shape `mask` finds along `r`, its direction unit length.
// `shapes` is taken, as Metal's trace() takes it, so the callers stay the
// same.
__device__ inline Hit trace(SceneStructure structures, Shapes /*shapes*/, Ray r,
                            serenity::contracts::RayMask mask) {
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return Hit{false, 0.0f, 0u, MeshPoint{}};
  }
  uint ignore = ignores_none;
  optixTraverse(traversed.handle, r.origin, r.direction, r.min_distance,
                r.max_distance, 0.0f, OptixVisibilityMask(traversed.bit),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0u, 1u, 0u, ignore);
  if (!optixHitObjectIsHit()) {
    return Hit{false, 0.0f, 0u, MeshPoint{}};
  }
  return Hit{true, optixHitObjectGetRayTmax(), optixHitObjectGetInstanceId(),
             hit_mesh_point()};
}

// Whether any shape `mask` finds lies along `r`: a shadow ray's question.
// It stops at the first such shape, whichever it is
// (TERMINATE_ON_FIRST_HIT), rather than searching for the nearest.
// `ignore` is the light a shadow ray aims at, which the occluders do not
// hold (above).
__device__ inline bool occluded(SceneStructure structures, Shapes /*shapes*/,
                                Ray r, serenity::contracts::RayMask mask,
                                uint ignore) {
  if (mask == serenity::contracts::RayMask::everything &&
      ignore != ignores_none) {
    // No shape can be skipped by name (above): the launch ends, and the
    // submission's next wait reports the fault (device/error.h).
    __trap();
  }
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return false;
  }
  uint unused = ignores_none;
  optixTraverse(
      traversed.handle, r.origin, r.direction, r.min_distance, r.max_distance,
      0.0f, OptixVisibilityMask(traversed.bit),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0u,
      1u, 0u, unused);
  return optixHitObjectIsHit();
}

}  // namespace shaders
}  // namespace serenity
