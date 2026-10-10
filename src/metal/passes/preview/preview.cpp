#include "metal/passes/preview/preview.h"

#include "metal/device/error.h"

namespace serenity::metal {

PreviewPass::PreviewPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("preview")) {}

void PreviewPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.radiance == nullptr) {
        throw Error("PreviewPass: the frame has no scene, no camera or no radiance image");
    }
    // Bindings match the kernel: the scene's block whole, and what changes
    // per frame beside it (metal/scene/scene_block.h).
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, 0);
    arguments->setAddress(resources.camera, 1);
    arguments->setResource(resources.acceleration, 2);
    arguments->setAddress(resources.scene, 3);
    arguments->setAddress(resources.glows, 4);       // the sphere lights' glows this frame
    arguments->setAddress(resources.transforms, 5);  // as this frame places the shapes
    arguments->setTexture(resources.radiance->gpuResourceID(), 0);
    encoder->setComputePipelineState(pipeline_.get());

    // As the test pattern: rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
