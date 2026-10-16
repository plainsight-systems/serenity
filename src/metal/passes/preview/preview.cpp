#include "metal/passes/preview/preview.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

PreviewPass::PreviewPass(const Library& library) : pipeline_(library.compute_pipeline("preview")) {}

void PreviewPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw MetalError("PreviewPass: the frame lacks one of its or radiance image");
    }
    // Bindings: passes/bindings.h, as preview.metal declares them: the
    // scene's block whole, and what changes per frame beside it
    // (metal/scene/scene_block.h).
    namespace binding = bindings::preview;
    MTL4::ArgumentTable* arguments = resources.arguments;
    bind_scene(resources, SceneCamera::bound, "PreviewPass");
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::radiance);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
