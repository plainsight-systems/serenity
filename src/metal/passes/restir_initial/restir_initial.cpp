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
    // The previous positions, written for a temporal merge, are placed by
    // last frame's transforms.
    if (resources.previous_positions != 0 && resources.last_transforms == 0) {
        throw MetalError("RestirInitialPass: the surfaces' previous positions are asked for without last frame's "
                         "transforms to place them by");
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
    // Metal requires every binding a kernel declares to be set. A graph with
    // no temporal merge has no previous positions nor last frame's
    // transforms, and its kernel neither writes nor reads them (history 0,
    // restir_initial.metal): the records and this frame's transforms stand
    // in those two places, untouched.
    const bool temporal = resources.previous_positions != 0;
    arguments->setAddress(temporal ? resources.previous_positions : resources.records, binding::previous_positions);
    arguments->setAddress(temporal ? resources.last_transforms : resources.transforms, binding::last_transforms);
    arguments->setAddress(output, binding::output);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
