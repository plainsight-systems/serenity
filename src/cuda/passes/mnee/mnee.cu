// The MNEE pass's program (mnee.h): its requests solved one thread each,
// their light added to ReSTIR PT's light buffer. Metal's mnee_solve
// (metal/passes/mnee/mnee.metal), as a ray generation program launched at
// the work list's capacity, a thread past the list's count returning: no
// launch takes its size from the GPU (docs/research/2026-10-19-cuda-
// backend.md, "Phase 4's design").

#include <optix.h>

#include "core/contracts/mnee_request.h"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/traced_module.cuh"
#include "cuda/integrator/mnee.cuh"
#include "cuda/integrator/path.cuh"
#include "cuda/scene/scene_block.cuh"

namespace serenity {
namespace shaders {

// Request n solved, its light added to its pixel's: three atomic adds,
// since the requests of one path's vertices share a pixel. A light not
// finite is counted (cuda/film/failure_count.h) and left out, as the shade
// leaves out a sample.
__device__ inline void solve_one() {
  namespace c = serenity::contracts;
  const uint n = optixGetLaunchIndex().x;
  // Past the capacity, none was written (contract 16).
  const uint count = min(launch.mnee_header[c::mnee_count_word],
                         launch.mnee_header[c::mnee_capacity_word]);
  bool failed = false;
  if (n < count) {
    const path::Scene scene{&launch.structures, launch.scene, launch.transforms,
                            launch.glows};
    const c::MneeRequest request = launch.mnee_requests[n];
    const float3 found = mnee::solve_request(scene, request);
    failed = !finite(found);
    if (!failed && any(found != 0.0f)) {
      float4& pixel = launch.light[request.pixel];
      atomicAdd(&pixel.x, found.x);
      atomicAdd(&pixel.y, found.y);
      atomicAdd(&pixel.z, found.z);
    }
  }
  count_non_finite(failed, launch.non_finite);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__mnee_solve() {
  serenity::shaders::solve_one();
}
