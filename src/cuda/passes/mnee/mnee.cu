// The MNEE pass's program (mnee.h) that traces: each converged walk's
// light (integrator/mnee.cuh, chain_light: the transfer matrix, the
// chain's transmittance, its visibility, the rays), added to ReSTIR PT's
// light buffer. The walks themselves are taken in rounds by the pass's
// kernels, which trace nothing (mnee_kernels.cu;
// docs/research/2026-10-21-tracing-divergence.md, section 4). A ray
// generation program launched at the converged list's capacity, a thread
// past its count returning: no launch takes its size from the GPU
// (docs/research/2026-10-19-cuda-backend.md, "Phase 4's design").

#include <optix.h>

#include "core/contracts/mnee_request.h"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/mnee_walk.h"
#include "cuda/frame/traced_module.cuh"
#include "cuda/integrator/mnee.cuh"
#include "cuda/integrator/path.cuh"
#include "cuda/scene/scene_block.cuh"

namespace serenity {
namespace shaders {

// Converged walk n's light, its request's chain rebuilt and its
// derivatives at the walk's points computed again (the same numbers the
// walk's last iteration had), added to its pixel's: three atomic adds,
// since the requests of one path's vertices share a pixel. A light not
// finite is counted (cuda/film/failure_count.h) and left out, as the shade
// leaves out a sample.
__device__ inline void finish_one() {
  const uint n = optixGetLaunchIndex().x;
  bool failed = false;
  if (n < launch.mnee_walk_header[serenity::cuda::mnee_converged_word]) {
    const serenity::cuda::MneeWalk walked = launch.mnee_converged[n];
    const path::Scene scene{&launch.structures, launch.scene, launch.transforms,
                            launch.glows};
    const mnee::Requested q =
        mnee::requested(scene, launch.mnee_requests[walked.request]);
    const mnee::Chain chain = mnee::chain_start(
        scene, q.here, q.origin, q.light_center, q.sample, q.seed);
    float3 p[mnee::max_interfaces];
#pragma unroll
    for (uint i = 0u; i < mnee::max_interfaces; ++i) {
      p[i] = to_float3(walked.p[i]);
    }
    mnee::Blocks blocks[mnee::max_interfaces];
    (void)mnee::constraint_blocks(chain.count, p, chain.casters, q.origin,
                                  chain.y, blocks);
    const float3 found =
        q.weight *
        mnee::chain_light(scene, q.here, q.origin, q.sample, chain, p, blocks);
    failed = !finite(found);
    if (!failed && any(found != 0.0f)) {
      float4& pixel = launch.light[launch.mnee_requests[walked.request].pixel];
      atomicAdd(&pixel.x, found.x);
      atomicAdd(&pixel.y, found.y);
      atomicAdd(&pixel.z, found.z);
    }
  }
  count_non_finite(failed, launch.non_finite);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__mnee_finish() {
  serenity::shaders::finish_one();
}
