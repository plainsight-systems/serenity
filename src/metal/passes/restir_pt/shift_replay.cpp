#include "metal/passes/restir_pt/shift_replay.h"

#include <string>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/reuse_constants.h"

namespace serenity::metal {

ShiftReplay::ShiftReplay(const Library& library,
                         const contracts::PathReuseSettings& settings,
                         Names names, std::string_view pass)
    : pair_(library.compute_pipeline(
          names.pair, {.group_threads = apple_group_threads,
                       .constants = reuse_constants(settings).get()})),
      grid_(library.compute_pipeline("restir_pt_pair_grids")),
      shift_(library.compute_pipeline(
          names.shift, {.group_threads = apple_group_threads,
                        .constants = reuse_constants(settings).get()})),
      merge_(library.compute_pipeline(
          names.merge, {.group_threads = apple_group_threads,
                        .constants = reuse_constants(settings).get()})),
      pass_(pass) {
  if (shift_->maxTotalThreadsPerThreadgroup() < contracts::replay_group) {
    throw MetalError(
        std::string(pass) +
        ": the shift's pipeline takes fewer threads a group than its grids' " +
        std::to_string(contracts::replay_group));
  }
}

void ShiftReplay::record(MTL4::ComputeCommandEncoder* encoder,
                         const FrameResources& resources) const {
  if (resources.reconnects == 0 || resources.replays == 0 ||
      resources.pair_counts == 0 ||
      resources.pair_grids[contracts::reconnect_list] == 0 ||
      resources.pair_grids[contracts::replay_list] == 0) {
    throw MetalError(std::string(pass_) +
                     ": the frame lacks its lists of pairs, their counts or "
                     "their grids");
  }
  namespace binding = bindings::restir_pt;
  MTL4::ArgumentTable* arguments = resources.arguments;
  arguments->setAddress(resources.reconnects, binding::reconnects);
  arguments->setAddress(resources.replays, binding::pairs);
  arguments->setAddress(resources.pair_counts, binding::counts);
  arguments->setAddress(resources.pair_grids[0], binding::dispatch);
  encoder->setComputePipelineState(pair_.get());
  dispatch_per_pixel(encoder, pair_.get(), resources.traced);
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(grid_.get());
  encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(shift_.get());
  for (const std::uint32_t list :
       {contracts::reconnect_list, contracts::replay_list}) {
    arguments->setAddress(list == contracts::reconnect_list
                              ? resources.reconnects
                              : resources.replays,
                          binding::pairs);
    arguments->setAddress(resources.pair_grids.at(list), binding::dispatch);
    encoder->dispatchThreads(resources.pair_grids.at(list));
  }
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(merge_.get());
  dispatch_per_pixel(encoder, merge_.get(), resources.traced);
}

}  // namespace serenity::metal
