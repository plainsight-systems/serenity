// ReSTIR PT's shade (restir_pt_shade.h): step 12 of passes/restir_pt/
// restir_pt.cuh, Metal's restir_pt_shade (metal/passes/restir_pt_shade/
// restir_pt_shade.metal), a kernel, one thread a traced pixel (restir_pt.cuh,
// Traced). The frame's light buffer, plus the initial reservoir's own
// estimate where no spatial pass merged one in, folded into the accumulated
// image as the path pass folds its own (passes/path/path.cu), whose mean is
// the frame's radiance. With the checkerboard a pixel is folded in every
// other frame, so it starts over in either of a history's first two frames,
// the first that trace it (contract 20); its mean is of the frames that
// traced it.

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/frame_launch.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"

namespace serenity {
namespace shaders {

__device__ inline void shade_pixel(const serenity::cuda::FrameLaunch& launch) {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel =
      traced.pixel(make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y));
  const bool inside = traced.inside(pixel);
  bool failed = false;
  if (inside) {
    const uint i = traced.slot(pixel);
    // neighbors and confidence_cap are 0 exactly in a graph without
    // restir_pt_spatial and restir_pt_temporal, whose merges add their
    // estimates to the light buffer (contracts/path_reuse_settings.h).
    const bool merged =
        settings.neighbors != 0u || settings.confidence_cap > 0.0f;
    const float3 sample =
        xyz(launch.light[i]) +
        (merged ? make_float3(0.0f)
                : restir_pt::estimate(launch.path_reservoirs[i]));
    failed = !finite(sample);
    // A pixel starts over in the image's first frame; with the
    // checkerboard in either of its first two, whichever traces it.
    const uint starting_frames = traced.checkerboard ? 2u : 1u;
    float4* const row = reinterpret_cast<float4*>(launch.accumulated.rows +
                                                  std::uint64_t{pixel.y} *
                                                      launch.accumulated.pitch);
    row[pixel.x] = accumulate(row[pixel.x], sample,
                              frame.accumulated_frames < starting_frames);
  }
  count_non_finite(failed, launch.non_finite);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_shade(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::shade_pixel(*launch);
}
