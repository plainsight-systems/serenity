#pragma once

// Axis: Acceleration (device half: the shapes' intersection program).
//
// The program OptiX runs for each box a ray enters (trace.cuh): the shape
// the box holds, by its structure's record (structure_block.h), tested
// exactly in its object space (shapes/shapes.cuh, intersect_shape), against
// the ray as OptiX has narrowed it, so only a hit nearer than the nearest
// so far is reported, as Metal's loop commits only a nearer one
// (metal/acceleration/trace.metal.h). The shape in payload 0 is never
// reported: a shadow ray's ignored shape (trace.cuh, occluded).
//
// Each pass that traces includes this file once, in its programs' `.cu`,
// whose module the program is then part of (cuda/device/pipeline.h: one
// module a pass), under the name its pipeline gives it. One program for
// every shape kind, as Metal's one loop: the kind is the record's, read in
// intersect_shape(). Its body is intersect_candidate(), in the namespace
// whose names it uses; the entry point OptiX names is one call to it.

#include <optix.h>

#include "cuda/acceleration/structure_block.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/shapes/shapes.cuh"

namespace serenity {
namespace shaders {

__device__ inline void intersect_candidate() {
  const auto& record =
      *reinterpret_cast<const serenity::cuda::StructureRecord*>(
          optixGetSbtDataPointer());
  const uint shape = record_shape(record, optixGetPrimitiveIndex());
  if (shape == optixGetPayload_0()) {
    return;
  }
  const Ray world =
      make_ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(),
               optixGetRayTmin(), optixGetRayTmax());
  const Crossing crossing = intersect_shape(
      Shapes{record.shapes, record.transforms, record.boxes}, shape, world);
  if (crossing.found) {
    optixReportIntersection(crossing.t, 0u);
  }
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __intersection__shape() {
  serenity::shaders::intersect_candidate();
}
