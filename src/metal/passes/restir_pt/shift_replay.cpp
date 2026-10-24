#include "metal/passes/restir_pt/shift_replay.h"

#include <string>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/restir_pt/reuse_constants.h"

namespace serenity::metal {

ShiftReplay::ShiftReplay(const Library& library,
                         const contracts::PathReuseSettings& settings,
                         const char* shift, const char* replay,
                         const char* merge, std::string_view pass)
    : shift_(library.compute_pipeline(
          shift, {.group_threads = apple_group_threads,
                  .constants = reuse_constants(settings).get()})),
      grid_(library.compute_pipeline("restir_pt_replay_grid")),
      replay_(library.compute_pipeline(
          replay, {.group_threads = apple_group_threads,
                   .constants = reuse_constants(settings).get()})),
      merge_(library.compute_pipeline(
          merge, {.group_threads = apple_group_threads,
                  .constants = reuse_constants(settings).get()})) {
  if (replay_->maxTotalThreadsPerThreadgroup() < contracts::replay_group) {
    throw MetalError(
        std::string(pass) +
        ": the replay's pipeline takes fewer threads a group than its grid's " +
        std::to_string(contracts::replay_group));
  }
}

void ShiftReplay::record(MTL4::ComputeCommandEncoder* encoder,
                         frame::Extent traced,
                         MTL::GPUAddress replay_grid) const {
  encoder->setComputePipelineState(shift_.get());
  dispatch_per_pixel(encoder, shift_.get(), traced);
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(grid_.get());
  encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(replay_.get());
  encoder->dispatchThreads(replay_grid);
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(merge_.get());
  dispatch_per_pixel(encoder, merge_.get(), traced);
}

}  // namespace serenity::metal
