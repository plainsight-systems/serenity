#include "metal/passes/path/path.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

PathPass::PathPass(const Library& library) : pipeline_(library.compute_pipeline("path_trace")) {}

void PathPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.accumulation == nullptr || resources.non_finite_counter == 0) {
        throw MetalError("PathPass: the frame lacks one of its accumulated image, or counter of samples not finite");
    }
    // Bindings: passes/bindings.h, as path.metal declares them: the scene's
    // block whole, and what changes per frame beside it
    // (metal/scene/scene_block.h).
    namespace binding = bindings::path;
    MTL4::ArgumentTable* arguments = resources.arguments;
    bind_scene(resources, SceneCamera::bound, "PathPass");
    arguments->setAddress(resources.non_finite_counter, binding::non_finite);
    arguments->setTexture(resources.accumulation->gpuResourceID(), binding::accumulated);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
