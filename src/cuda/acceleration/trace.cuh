#pragma once

// Axis: Acceleration (device half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any shape blocks a shadow
// ray, among the shapes the ray's mask finds (contract 19,
// core/contracts/ray_mask.h): Metal's trace() and occluded()
// (metal/acceleration/trace.metal.h), with the same signatures, as OptiX
// answers them (docs/research/2026-10-19-cuda-backend.md, "Phase 3's
// design"). Kind-blind: it names no shape kind.
//
// Metal's hardware walks a structure of bounding boxes and hands each box
// the ray enters to the shader's loop, which runs the shape's exact test
// and commits a nearer hit. Here OptiX walks the structure and runs the
// shapes' intersection program for each box (intersection.cuh), which runs
// the same exact test, against the ray as OptiX has narrowed it to the
// nearest hit so far, and reports what it finds; optixTraverse() returns
// the nearest in a hit object, which trace() reads, running no
// closest-hit or miss program. So a ray finds the same shape at the same t
// as Metal's loop would, up to each GPU's arithmetic.
//
// One level (scene_acceleration.h): a ray traces the structure its mask
// names, with that structure's hit record (structure_block.h), whose table
// names the shape a box holds; the program reads it, and trace() reads it
// again for the shape it reports. What a hit reports is the shape and t;
// where on it, in the world, is surface_interaction()'s (shapes/
// shapes.cuh).
//
// The mask is an argument, not a bound value: every site passes a constant
// (I.4: a RayMask, no integer), so the compiler picks the structure where
// it is traced, and the threads of a warp at one site trace the same one
// (GPU.4).
//
// Payload 0 is the shape the intersection program ignores, none for
// trace(); the pipeline gives each trace that one value
// (passes/path/path.cpp). Any-hit is off on the geometry and the ray: the
// intersection program decides every hit.

#include <optix.h>

#include "core/contracts/ray_mask.h"
#include "cuda/acceleration/structure_block.h"
#include "cuda/math/ray.cuh"
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

// What a trace carrying one mask traverses: the structure's handle and the
// hit record it is traced with; `empty`, the mask finds no shape in this
// scene (occluders, in a scene of lights alone), and no trace is made.
struct MaskedStructure {
  OptixTraversableHandle handle;
  uint record;
  bool empty;
};

__device__ inline MaskedStructure masked(SceneStructure structures,
                                         serenity::contracts::RayMask mask) {
  if (mask == serenity::contracts::RayMask::everything) {
    return MaskedStructure{structures->everything,
                           serenity::cuda::everything_record, false};
  }
  return MaskedStructure{structures->occluders,
                         serenity::cuda::occluders_record,
                         structures->occluder_count == 0u};
}

// The shape box `primitive` holds, by its structure's record.
__device__ inline uint record_shape(
    const serenity::cuda::StructureRecord& record, uint primitive) {
  return record.table == nullptr ? primitive : record.table[primitive];
}

struct Hit {
  bool found;
  float t;  // the ray's parameter: its distance, the direction being unit
  uint primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
};

// The nearest shape `mask` finds along `r`, its direction unit length. The
// shapes are the hit records' (structure_block.h); `shapes` is taken, as
// Metal's trace() takes it, so the callers stay the same.
__device__ inline Hit trace(SceneStructure structures, Shapes /*shapes*/, Ray r,
                            serenity::contracts::RayMask mask) {
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return Hit{false, 0.0f, 0u};
  }
  uint ignore = ignores_none;
  optixTraverse(traversed.handle, r.origin, r.direction, r.min_distance,
                r.max_distance, 0.0f, OptixVisibilityMask(0xffu),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, traversed.record, 1u, 0u,
                ignore);
  if (!optixHitObjectIsHit()) {
    return Hit{false, 0.0f, 0u};
  }
  const auto* record = reinterpret_cast<const serenity::cuda::StructureRecord*>(
      optixHitObjectGetSbtDataPointer());
  return Hit{true, optixHitObjectGetRayTmax(),
             record_shape(*record, optixHitObjectGetPrimitiveIndex())};
}

// Whether any shape `mask` finds but `ignore` lies along `r`: a shadow
// ray's question. It stops at the first such shape, whichever it is
// (TERMINATE_ON_FIRST_HIT), rather than searching for the nearest. A shadow
// ray toward a light carries the occluders, which hold no light, so its
// light's shape is not among them at all; `ignore` stays for a ray that
// carries everything.
__device__ inline bool occluded(SceneStructure structures, Shapes /*shapes*/,
                                Ray r, serenity::contracts::RayMask mask,
                                uint ignore) {
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return false;
  }
  optixTraverse(
      traversed.handle, r.origin, r.direction, r.min_distance, r.max_distance,
      0.0f, OptixVisibilityMask(0xffu),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      traversed.record, 1u, 0u, ignore);
  return optixHitObjectIsHit();
}

}  // namespace shaders
}  // namespace serenity
