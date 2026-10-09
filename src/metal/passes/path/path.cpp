#include "metal/passes/path/path.h"

#include "metal/acceleration/primitives.h"
#include "metal/device/error.h"

namespace serenity::metal {

PathPass::PathPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("path_trace")) {}

void PathPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == nullptr || resources.acceleration == nullptr || resources.camera == 0 ||
        resources.accumulation == nullptr || resources.non_finite_counter == 0) {
        throw Error("PathPass: the frame has no scene, no camera or no accumulated image");
    }
    const SceneBuffers::Addresses& scene = *resources.scene;

    // Bindings match path.metal.
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, 0);
    arguments->setAddress(resources.camera, 1);
    arguments->setResource(resources.acceleration->resource(), 2);
    arguments->setAddress(scene.environment, 3);
    arguments->setAddress(scene.textures, 4);
    arguments->setAddress(scene.checkers, 5);
    arguments->setAddress(scene.materials, 6);
    arguments->setAddress(scene.rough, 7);
    arguments->setAddress(scene.dielectrics, 8);
    arguments->setAddress(scene.conductors, 9);
    arguments->setAddress(scene.emissives, 10);
    arguments->setAddress(scene.shapes, 11);
    arguments->setAddress(scene.spheres, 12);
    arguments->setAddress(scene.boxes, 13);
    arguments->setAddress(scene.light_records, 14);
    arguments->setAddress(scene.sphere_lights, 15);
    arguments->setAddress(scene.light_counts, 16);
    arguments->setAddress(resources.non_finite_counter, 17);
    arguments->setTexture(resources.accumulation->gpuResourceID(), 0);
    arguments->setTexture(resources.target->gpuResourceID(), 1);
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
