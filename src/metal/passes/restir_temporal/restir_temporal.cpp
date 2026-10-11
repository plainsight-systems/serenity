#include "metal/passes/restir_temporal/restir_temporal.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

RestirTemporalPass::RestirTemporalPass(const Library& library, ReservoirOutput output)
    : pipeline_(library.compute_pipeline("restir_temporal")), output_(output) {}

void RestirTemporalPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.reuse == 0 || resources.records == 0 || resources.previous_positions == 0 ||
        resources.last_records == 0 || resources.last_final_reservoirs == 0 || resources.working_reservoirs == 0 ||
        resources.last_transforms == 0 || resources.last_glows == 0) {
        throw MetalError("RestirTemporalPass: the frame lacks one of its scene, transforms, glows, camera, reuse "
                         "constants, surface records, previous positions, last frame's records, final reservoirs, "
                         "transforms or glows, or the initial reservoirs");
    }
    // Where the graph says the merged reservoirs go (frame_resources.h): in
    // place when a spatial merge follows.
    const MTL::GPUAddress output =
        output_ == ReservoirOutput::working ? resources.working_reservoirs : resources.final_reservoirs;
    if (output == 0) {
        throw MetalError("RestirTemporalPass: the frame has no final reservoirs to write");
    }
    // Bindings: passes/bindings.h, as restir_temporal.metal declares them.
    namespace binding = bindings::restir_temporal;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.camera, binding::camera);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.reuse, binding::reuse);
    arguments->setAddress(resources.records, binding::records);
    arguments->setAddress(resources.previous_positions, binding::previous_positions);
    arguments->setAddress(resources.last_records, binding::last_records);
    arguments->setAddress(resources.last_final_reservoirs, binding::last_final);
    arguments->setAddress(resources.working_reservoirs, binding::input);
    arguments->setAddress(output, binding::output);
    arguments->setAddress(resources.last_glows, binding::last_glows);
    arguments->setAddress(resources.last_transforms, binding::last_transforms);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
