#include "metal/passes/display/display.h"

#include "metal/device/error.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

DisplayPass::DisplayPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("display")) {}

void DisplayPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw Error("DisplayPass: the frame has no radiance image");
    }
    // Bindings: passes/bindings.h, as display.metal declares them.
    resources.arguments->setAddress(resources.constants, bindings::display::constants);
    resources.arguments->setTexture(resources.radiance->gpuResourceID(), bindings::display::radiance);
    resources.arguments->setTexture(resources.target->gpuResourceID(), bindings::display::target);
    encoder->setComputePipelineState(pipeline_.get());

    // Rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
