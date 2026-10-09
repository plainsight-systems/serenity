#include "metal/passes/test_pattern/test_pattern.h"

namespace serenity::metal {

TestPatternPass::TestPatternPass(const Device&, const Library& library)
    : pipeline_(library.compute_pipeline("test_pattern")) {}

void TestPatternPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    // Bindings match test_pattern.metal: buffer 0, texture 0.
    resources.arguments->setAddress(resources.constants, 0);
    resources.arguments->setTexture(resources.target->gpuResourceID(), 0);
    const frame::Extent size = resources.size;
    encoder->setComputePipelineState(pipeline_.get());

    // Rows of the execution width, as many rows as the threadgroup allows:
    // adjacent threads write adjacent pixels of a row (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(size.width, size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
