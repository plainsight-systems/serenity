#include "metal/passes/restir_initial/restir_initial.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

RestirInitialPass::RestirInitialPass(const Library& library, ReservoirOutput output)
    : pipeline_(library.compute_pipeline("restir_initial")), output_(output) {}

void RestirInitialPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.reuse == 0 || resources.records == 0 || resources.path_states == 0) {
        throw MetalError("RestirInitialPass: the frame lacks one of its scene, transforms, glows, camera, reuse "
                         "constants, surface records or path states");
    }
    // Where the graph says the reservoirs go (frame_resources.h).
    const MTL::GPUAddress output =
        output_ == ReservoirOutput::working ? resources.working_reservoirs : resources.final_reservoirs;
    if (output == 0) {
        throw MetalError(output_ == ReservoirOutput::working
                             ? "RestirInitialPass: the frame has no working reservoirs to write"
                             : "RestirInitialPass: the frame has no final reservoirs to write");
    }
    // The previous positions are written for a temporal merge, which this
    // backend does not run yet (metal/frame/renderer.h): asked for, they
    // are refused, not silently left unwritten.
    if (resources.previous_positions != 0) {
        throw MetalError("RestirInitialPass: the surfaces' previous positions are written for restir_temporal, "
                         "which this backend does not implement yet");
    }
    // Bindings: passes/bindings.h, as restir_initial.metal declares them.
    namespace binding = bindings::restir_initial;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.camera, binding::camera);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.reuse, binding::reuse);
    arguments->setAddress(resources.records, binding::records);
    arguments->setAddress(resources.path_states, binding::path_states);
    arguments->setAddress(output, binding::output);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
