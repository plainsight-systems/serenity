#include "metal/passes/test_pattern/test_pattern.h"

#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

TestPatternPass::TestPatternPass(const Library& library) : pipeline_(library.compute_pipeline("test_pattern")) {}

void TestPatternPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    // Bindings: passes/bindings.h, as test_pattern.metal declares them.
    resources.arguments->setAddress(resources.constants, bindings::test_pattern::constants);
    resources.arguments->setTexture(resources.target->gpuResourceID(), bindings::test_pattern::target);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
