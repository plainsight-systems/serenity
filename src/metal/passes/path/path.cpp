#include "metal/passes/path/path.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

PathPass::PathPass(const Library& library, bool mnee)
    : pipeline_(library.compute_pipeline(mnee ? "path_trace_mnee" : "path_trace")) {
    if (mnee) {
        mnee_.emplace(library);
    }
}

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
    if (mnee_) {
        if (resources.mnee_requests == 0 || resources.mnee_header == 0) {
            throw MetalError("PathPass: the frame lacks MNEE's work list");
        }
        arguments->setAddress(resources.mnee_requests, binding::mnee_requests);
        arguments->setAddress(resources.mnee_header, binding::mnee_header);
    }
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
    if (mnee_) {
        // The paths' MNEE requests solved and folded into the image.
        mnee_->record(encoder, resources, MneePass::Into::accumulated_image);
    }
}

}  // namespace serenity::metal
