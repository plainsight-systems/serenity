#include "metal/passes/restir_pt_shade/restir_pt_shade.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/reuse_constants.h"

namespace serenity::metal {

RestirPtShadePass::RestirPtShadePass(
    const Library& library, const contracts::PathReuseSettings& settings)
    : pipeline_(library.compute_pipeline(
          "restir_pt_shade", {.group_threads = apple_group_threads,
                              .constants = reuse_constants(settings).get()})) {}

void RestirPtShadePass::record(MTL4::ComputeCommandEncoder* encoder,
                               const FrameResources& resources) const {
  if (resources.path_reservoirs == 0 || resources.light == 0 ||
      resources.accumulation == nullptr || resources.non_finite_counter == 0) {
    throw MetalError(
        "RestirPtShadePass: the frame lacks one of ReSTIR PT's path "
        "reservoirs, light "
        "buffer, the accumulated image, or the counter of samples not finite");
  }
  namespace binding = bindings::restir_pt;
  MTL4::ArgumentTable* arguments = resources.arguments;
  arguments->setAddress(resources.constants, binding::constants);
  arguments->setAddress(resources.path_reservoirs, binding::reservoirs);
  arguments->setAddress(resources.light, binding::light);
  arguments->setAddress(resources.non_finite_counter, binding::non_finite);
  arguments->setTexture(resources.accumulation->gpuResourceID(),
                        binding::accumulated);
  encoder->setComputePipelineState(pipeline_.get());
  dispatch_per_pixel(encoder, pipeline_.get(),
                     resources.traced);  // a thread a traced pixel
}

}  // namespace serenity::metal
