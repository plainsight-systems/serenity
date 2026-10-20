// ReSTIR PT's temporal reuse's kernel (restir_pt_temporal.h) that traces
// nothing: step 7, the Talbot merge, Metal's restir_pt_temporal
// (metal/passes/restir_pt_temporal/restir_pt_temporal.metal), taking the
// frame's launch parameters by address (frame/frame_launch.h).

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/passes/restir_pt_temporal/temporal_slots.cuh"

namespace serenity {
namespace shaders {

// Step 7: the Talbot merge, one thread a traced pixel, into the reservoirs
// in place where a spatial reuse follows, else into the frame's final ones,
// its estimate then added to the light buffer (its own pixel's alone).
__device__ inline void temporal_merge(
    const serenity::cuda::FrameLaunch& launch) {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel =
      traced.pixel(make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y));
  if (!traced.inside(pixel)) {
    return;
  }
  const uint i = traced.slot(pixel);
  const serenity::contracts::PathReservoir own = launch.path_reservoirs[i];
  const uint j = launch.previous_pixels[i];
  restir_pt::Merged merged{own, restir_pt::estimate(own)};
  if (j != restir_pt::no_pixel) {
    // The cap at last's pixel, lowered by last frame's duplication map in a
    // graph that has one (cap_exponent 0 otherwise, its map unread).
    const float cap = settings.cap_exponent > 0.0f
                          ? path_merge::duplication_cap(
                                settings.confidence_cap, settings.cap_floor,
                                settings.cap_exponent, launch.duplication[j])
                          : settings.confidence_cap;
    PathNumbers numbers = purpose_numbers(
        path_numbers(pixel, frame.frame_index), restir_pt::temporal_pick_first);
    merged = restir_pt::temporal_merge(
        own, launch.last_path_finals[j],
        launch.shifts[i * temporal::slots + temporal::last_here],
        launch.shifts[i * temporal::slots + temporal::own_there], cap,
        next_numbers2(numbers));
  }
  // neighbors is 0 exactly in a graph without restir_pt_spatial
  // (contracts/path_reuse_settings.h).
  if (settings.neighbors != 0u) {
    launch.path_reservoirs[i] = merged.reservoir;
    return;
  }
  launch.path_finals[i] = merged.reservoir;
  launch.light[i] += make_float4(merged.estimate, 0.0f);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_temporal(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::temporal_merge(*launch);
}
