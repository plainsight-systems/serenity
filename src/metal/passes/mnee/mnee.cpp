#include "metal/passes/mnee/mnee.h"

#include <cstdint>
#include <string>

#include "core/contracts/mnee_request.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

namespace {

// A walk count's address in the walks' header (contract 16).
MTL::GPUAddress count_word(MTL::GPUAddress header, std::uint32_t word) {
  return header + MTL::GPUAddress{word} * sizeof(std::uint32_t);
}

}  // namespace

MneePass::MneePass(const Library& library)
    : grid_(library.compute_pipeline("mnee_grid")),
      walk_grid_(library.compute_pipeline("mnee_walk_grid")),
      start_(library.compute_pipeline("mnee_walk_start",
                                      {.group_threads = apple_group_threads})),
      round_(library.compute_pipeline("mnee_walk_round",
                                      {.group_threads = apple_group_threads})),
      finish_(library.compute_pipeline(
          "mnee_finish", {.group_threads = apple_group_threads})) {
  // Each list's threadgroup, as the grid kernels write it (contract 16).
  for (const MTL::ComputePipelineState* pipeline :
       {start_.get(), round_.get(), finish_.get()}) {
    if (pipeline->maxTotalThreadsPerThreadgroup() <
        contracts::mnee_solve_group) {
      throw MetalError(
          "MneePass: a walk's pipeline takes fewer threads a group than its "
          "grid's " +
          std::to_string(contracts::mnee_solve_group));
    }
  }
}

void MneePass::record(MTL4::ComputeCommandEncoder* encoder,
                      const FrameResources& resources) const {
  if (resources.mnee_requests == 0 || resources.mnee_header == 0 ||
      resources.mnee_walks[0] == 0 || resources.mnee_walks[1] == 0 ||
      resources.mnee_converged == 0 || resources.mnee_walk_header == 0 ||
      resources.non_finite_counter == 0 || resources.mnee_lost_counter == 0 ||
      resources.light == 0) {
    throw MetalError(
        "MneePass: the frame lacks one of its MNEE work list, walk lists, "
        "counter of samples not finite or of lost requests, or the light "
        "buffer it adds to");
  }
  namespace binding = bindings::mnee;
  MTL4::ArgumentTable* arguments = resources.arguments;
  bind_scene(resources, SceneCamera::none, "MneePass");
  arguments->setAddress(resources.non_finite_counter, binding::non_finite);
  arguments->setAddress(resources.mnee_requests, binding::requests);
  arguments->setAddress(resources.mnee_header, binding::header);
  arguments->setAddress(resources.mnee_grid, binding::grid);
  arguments->setAddress(resources.light, binding::light);
  arguments->setAddress(resources.mnee_lost_counter, binding::lost);
  arguments->setAddress(resources.mnee_converged, binding::converged);
  arguments->setAddress(
      count_word(resources.mnee_walk_header, contracts::mnee_converged_word),
      binding::converged_count);
  // Every dispatch reads what the one before wrote (GPU.8).
  const auto one_thread = [&](const MTL::ComputePipelineState* pipeline) {
    barrier_after_dispatch(encoder);
    encoder->setComputePipelineState(pipeline);
    encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
  };
  const auto over_grid = [&](const MTL::ComputePipelineState* pipeline) {
    barrier_after_dispatch(encoder);
    encoder->setComputePipelineState(pipeline);
    encoder->dispatchThreads(resources.mnee_grid);
  };
  one_thread(grid_.get());
  arguments->setAddress(resources.mnee_walks[0], binding::walks_out);
  arguments->setAddress(count_word(resources.mnee_walk_header, 0),
                        binding::walk_count_out);
  over_grid(start_.get());
  for (std::uint32_t round = 1; round < contracts::mnee_rounds; ++round) {
    // Round `round` reads the list the round before wrote and writes the
    // other (contract 16).
    const std::uint32_t from = (round - 1) % contracts::mnee_walk_lists;
    const std::uint32_t into = round % contracts::mnee_walk_lists;
    arguments->setAddress(count_word(resources.mnee_walk_header, from),
                          binding::walk_count);
    one_thread(walk_grid_.get());
    arguments->setAddress(resources.mnee_walks.at(from), binding::walks_in);
    arguments->setAddress(resources.mnee_walks.at(into), binding::walks_out);
    arguments->setAddress(count_word(resources.mnee_walk_header, into),
                          binding::walk_count_out);
    over_grid(round_.get());
  }
  arguments->setAddress(
      count_word(resources.mnee_walk_header, contracts::mnee_converged_word),
      binding::walk_count);
  one_thread(walk_grid_.get());
  arguments->setAddress(resources.mnee_converged, binding::walks_in);
  over_grid(finish_.get());
}

}  // namespace serenity::metal
