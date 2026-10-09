#include "metal/passes/display/display.h"

#include "metal/device/error.h"

namespace serenity::metal {

DisplayPass::DisplayPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("display")) {}

void DisplayPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw Error("DisplayPass: the frame has no radiance image");
    }
    // Bindings match display.metal.
    resources.arguments->setAddress(resources.constants, 0);
    resources.arguments->setTexture(resources.radiance->gpuResourceID(), 0);
    resources.arguments->setTexture(resources.target->gpuResourceID(), 1);
    encoder->setComputePipelineState(pipeline_.get());

    // Rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
