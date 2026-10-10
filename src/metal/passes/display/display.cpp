#include "metal/passes/display/display.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

DisplayPass::DisplayPass(const Library& library) : pipeline_(library.compute_pipeline("display")) {}

void DisplayPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw MetalError("DisplayPass: the frame has no radiance image");
    }
    // Bindings: passes/bindings.h, as display.metal declares them.
    resources.arguments->setAddress(resources.constants, bindings::display::constants);
    resources.arguments->setTexture(resources.radiance->gpuResourceID(), bindings::display::radiance);
    resources.arguments->setTexture(resources.target->gpuResourceID(), bindings::display::target);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
