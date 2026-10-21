#pragma once

// Axis: Integrator (device half: MNEE's seed, as OptiX walks it).
//
// The intersection program OptiX runs for each box a seed's segment enters
// (integrator/mnee.cuh, seed_segment), the occluders' boxes under the
// seed's instance: the box's shape, by the seed's record
// (acceleration/structure_block.h), handed to seed_candidate() with
// the seed the ray generation program keeps, whose address payload words 1
// and 2 carry; the shape in word 0, the light's, skipped. Where the seed
// stops, the program reports a hit, at the middle of the segment's
// interval, a place OptiX takes (any t within it would do: the hit is read
// only as the walk's end); otherwise it reports nothing and OptiX goes on.
//
// Each tracing pass includes this file once in its programs' `.cu`
// (frame/traced_module.cuh), under the name the pass's pipeline gives it:
// the one intersection program the backend has, the shapes being
// triangles.

#include <optix.h>

#include "cuda/acceleration/structure_block.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/integrator/mnee.cuh"

namespace serenity {
namespace shaders {
namespace mnee {

// The shape box `primitive` of the occluders' boxes holds, by the table in
// the seed's record (cuda/acceleration/structure_block.h).
__device__ inline uint record_shape(
    const serenity::cuda::StructureRecord& record, uint primitive) {
  return record.table[primitive];
}

__device__ inline void seed_box() {
  const auto& record =
      *reinterpret_cast<const serenity::cuda::StructureRecord*>(
          optixGetSbtDataPointer());
  const uint shape = record_shape(record, optixGetPrimitiveIndex());
  if (shape == optixGetPayload_0()) {
    return;
  }
  const unsigned long long address =
      (static_cast<unsigned long long>(optixGetPayload_1()) << 32u) |
      optixGetPayload_2();
  SeedQuery& query = *reinterpret_cast<SeedQuery*>(address);
  if (seed_candidate(query, shape)) {
    optixReportIntersection(0.5f * (optixGetRayTmin() + optixGetRayTmax()), 0u);
  }
}

}  // namespace mnee
}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __intersection__seed() {
  serenity::shaders::mnee::seed_box();
}
