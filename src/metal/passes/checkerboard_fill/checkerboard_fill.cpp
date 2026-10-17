#include "metal/passes/checkerboard_fill/checkerboard_fill.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

CheckerboardFillPass::CheckerboardFillPass(const Library& library)
    : pipeline_(library.compute_pipeline("checkerboard_fill")) {}

void CheckerboardFillPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.checkerboard_constants == 0 || resources.accumulation == nullptr || resources.depth == nullptr ||
        resources.motion == nullptr || resources.light_radiance == nullptr || resources.previous_light == nullptr ||
        resources.history_light == nullptr) {
        throw MetalError("CheckerboardFillPass: the frame lacks one of the fill's constants, the accumulated image, "
                         "its depth or motion, its hand-off, or either frame's history");
    }
    // Bindings: passes/bindings.h, as checkerboard_fill.metal declares them.
    namespace binding = bindings::checkerboard_fill;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.checkerboard_constants, binding::fill_constants);
    arguments->setTexture(resources.accumulation->gpuResourceID(), binding::accumulated);
    arguments->setTexture(resources.previous_light->gpuResourceID(), binding::previous_light);
    arguments->setTexture(resources.depth->gpuResourceID(), binding::depth);
    arguments->setTexture(resources.motion->gpuResourceID(), binding::motion);
    arguments->setTexture(resources.light_radiance->gpuResourceID(), binding::light);
    arguments->setTexture(resources.history_light->gpuResourceID(), binding::history);
    encoder->setComputePipelineState(pipeline_.get());
    // One thread a slot of the packed grid, its two pixels (the header):
    // the grid ReSTIR PT's kernels ran over (frame_resources.h, traced).
    dispatch_per_pixel(encoder, pipeline_.get(), resources.traced);
}

}  // namespace serenity::metal
