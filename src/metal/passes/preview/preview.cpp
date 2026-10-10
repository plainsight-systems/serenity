#include "metal/passes/preview/preview.h"

#include "metal/device/error.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

PreviewPass::PreviewPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("preview")) {}

void PreviewPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.radiance == nullptr) {
        throw Error("PreviewPass: the frame lacks one of its scene, transforms, glows, camera, or radiance image");
    }
    // Bindings: passes/bindings.h, as preview.metal declares them: the
    // scene's block whole, and what changes per frame beside it
    // (metal/scene/scene_block.h).
    namespace binding = bindings::preview;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.camera, binding::camera);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);  // the sphere lights' glows this frame
    arguments->setAddress(resources.transforms, binding::transforms);  // as this frame places the shapes
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::radiance);
    encoder->setComputePipelineState(pipeline_.get());

    // As the test pattern: rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
