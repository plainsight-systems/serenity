#include "metal/passes/path/path.h"

#include "metal/device/error.h"

namespace serenity::metal {

PathPass::PathPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("path_trace")) {}

void PathPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.accumulation == nullptr || resources.non_finite_counter == 0 || resources.radiance == nullptr) {
        throw Error("PathPass: the frame has no scene, no camera, no accumulated image or no radiance image");
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
    arguments->setAddress(resources.non_finite_counter, 6);
    arguments->setTexture(resources.accumulation->gpuResourceID(), 0);
    arguments->setTexture(resources.radiance->gpuResourceID(), 1);
    encoder->setComputePipelineState(pipeline_.get());

    // Rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
