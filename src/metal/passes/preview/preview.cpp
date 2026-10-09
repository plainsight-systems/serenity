#include "metal/passes/preview/preview.h"

#include "metal/device/error.h"

namespace serenity::metal {

PreviewPass::PreviewPass(const Device&, const Library& library) : pipeline_(library.compute_pipeline("preview")) {}

void PreviewPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == nullptr || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.radiance == nullptr) {
        throw Error("PreviewPass: the frame has no scene, no camera or no radiance image");
    }
    const SceneBuffers::Addresses& scene = *resources.scene;

    // Bindings match preview.metal.
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
    arguments->setTexture(resources.radiance->gpuResourceID(), 0);
    encoder->setComputePipelineState(pipeline_.get());

    // As the test pattern: rows of the execution width (GPU.2).
    const NS::UInteger width = pipeline_->threadExecutionWidth();
    const NS::UInteger rows = pipeline_->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size(resources.size.width, resources.size.height, 1), MTL::Size(width, rows, 1));
}

}  // namespace serenity::metal
