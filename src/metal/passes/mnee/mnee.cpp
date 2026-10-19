#include "metal/passes/mnee/mnee.h"

#include <string>

#include "core/contracts/mnee_request.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

MneePass::MneePass(const Library& library)
    : grid_(library.compute_pipeline("mnee_grid")),
      solve_(library.compute_pipeline("mnee_solve")) {
  // The solve's threadgroup, as mnee.metal's grid writes it (contract 16).
  if (solve_->maxTotalThreadsPerThreadgroup() < contracts::mnee_solve_group) {
    throw MetalError(
        "MneePass: the solve's pipeline takes fewer threads a group than its "
        "grid's " +
        std::to_string(contracts::mnee_solve_group));
  }
}

void MneePass::record(MTL4::ComputeCommandEncoder* encoder,
                      const FrameResources& resources) const {
  if (resources.mnee_requests == 0 || resources.mnee_header == 0 ||
      resources.non_finite_counter == 0 || resources.mnee_lost_counter == 0 ||
      resources.light == 0) {
    throw MetalError(
        "MneePass: the frame lacks one of its MNEE work list, counter of "
        "samples not finite or of "
        "lost requests, or the light buffer it adds to");
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
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(grid_.get());
  encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
  barrier_after_dispatch(encoder);
  encoder->setComputePipelineState(solve_.get());
  encoder->dispatchThreads(resources.mnee_grid);
}

}  // namespace serenity::metal
