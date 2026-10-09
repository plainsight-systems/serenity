#include "metal/passes/path/path.h"

#include "metal/device/error.h"

namespace serenity::metal {

PathPass::PathPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("path_trace")) {}

void PathPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == nullptr || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.accumulation == nullptr || resources.non_finite_counter == 0 || resources.radiance == nullptr) {
        throw Error("PathPass: the frame has no scene, no camera, no accumulated image or no radiance image");
    }
    const SceneBuffers::Addresses& scene = *resources.scene;

    // Bindings match path.metal.
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, 0);
    arguments->setAddress(resources.camera, 1);
    arguments->setResource(resources.acceleration, 2);
    arguments->setAddress(scene.environment, 3);
    arguments->setAddress(scene.textures, 4);
    arguments->setAddress(scene.checkers, 5);
    arguments->setAddress(scene.materials, 6);
    arguments->setAddress(scene.rough, 7);
    arguments->setAddress(scene.dielectrics, 8);
    arguments->setAddress(scene.conductors, 9);
    arguments->setAddress(resources.glows, 10);  // the sphere lights' glows this frame
    arguments->setAddress(scene.shapes, 11);
    arguments->setAddress(resources.transforms, 12);  // as this frame places the shapes
    arguments->setAddress(scene.boxes, 13);
    arguments->setAddress(scene.light_records, 14);
    arguments->setAddress(scene.shape_lights, 15);
    arguments->setAddress(scene.sphere_lights, 16);
    arguments->setAddress(scene.light_counts, 17);
    arguments->setAddress(resources.non_finite_counter, 18);
    arguments->setTexture(resources.accumulation->gpuResourceID(), 0);
    arguments->setTexture(resources.radiance->gpuResourceID(), 1);
    encoder->setComputePipelineState(pipeline_.get());

    // The previous frame's dispatch wrote the accumulated image this one
    // reads; Metal 4 does not order them unless asked (path.h).
    encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

    // Rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
