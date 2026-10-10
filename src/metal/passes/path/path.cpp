#include "metal/passes/path/path.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

PathPass::PathPass(const Library& library) : pipeline_(library.compute_pipeline("path_trace")) {}

void PathPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.accumulation == nullptr || resources.non_finite_counter == 0) {
        throw MetalError("PathPass: the frame lacks one of its scene, transforms, glows, camera, accumulated image, "
                         "or counter of samples not finite");
    }
    // Bindings: passes/bindings.h, as path.metal declares them: the scene's
    // block whole, and what changes per frame beside it
    // (metal/scene/scene_block.h).
    namespace binding = bindings::path;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.camera, binding::camera);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);  // the sphere lights' glows this frame
    arguments->setAddress(resources.transforms, binding::transforms);  // as this frame places the shapes
    arguments->setAddress(resources.non_finite_counter, binding::non_finite);
    arguments->setTexture(resources.accumulation->gpuResourceID(), binding::accumulated);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
