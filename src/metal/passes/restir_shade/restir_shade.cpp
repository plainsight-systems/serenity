#include "metal/passes/restir_shade/restir_shade.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

RestirShadePass::RestirShadePass(const Library& library) : pipeline_(library.compute_pipeline("restir_shade")) {}

void RestirShadePass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.reuse == 0 ||
        resources.records == 0 || resources.path_states == 0 || resources.final_reservoirs == 0 ||
        resources.accumulation == nullptr || resources.non_finite_counter == 0) {
        throw MetalError("RestirShadePass: the frame lacks one of its scene, transforms, glows, reuse constants, "
                         "surface records, path states, final reservoirs, accumulated image, or counter of samples "
                         "not finite");
    }
    // Bindings: passes/bindings.h, as restir_shade.metal declares them.
    namespace binding = bindings::restir_shade;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.reuse, binding::reuse);
    arguments->setAddress(resources.records, binding::records);
    arguments->setAddress(resources.path_states, binding::path_states);
    arguments->setAddress(resources.final_reservoirs, binding::final_reservoirs);
    arguments->setAddress(resources.non_finite_counter, binding::non_finite);
    arguments->setTexture(resources.accumulation->gpuResourceID(), binding::accumulated);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
