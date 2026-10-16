#include "metal/passes/features/features.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

FeaturesPass::FeaturesPass(const Library& library) : pipeline_(library.compute_pipeline("features")) {}

void FeaturesPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.feature_constants == 0 || resources.feature_last_transforms == 0 || resources.albedo == nullptr ||
        resources.normal == nullptr || resources.depth == nullptr || resources.motion == nullptr) {
        throw MetalError("FeaturesPass: the frame lacks one of its feature constants, last frame's transforms or "
                         "feature images");
    }
    // Bindings: passes/bindings.h, as features.metal declares them.
    namespace binding = bindings::features;
    MTL4::ArgumentTable* arguments = resources.arguments;
    bind_scene(resources, SceneCamera::bound, "FeaturesPass");
    arguments->setAddress(resources.feature_constants, binding::feature_constants);
    arguments->setAddress(resources.feature_last_transforms, binding::last_transforms);
    arguments->setTexture(resources.albedo->gpuResourceID(), binding::albedo);
    arguments->setTexture(resources.normal->gpuResourceID(), binding::normal);
    arguments->setTexture(resources.depth->gpuResourceID(), binding::depth);
    arguments->setTexture(resources.motion->gpuResourceID(), binding::motion);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
