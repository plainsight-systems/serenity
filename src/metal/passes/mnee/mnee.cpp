#include "metal/passes/mnee/mnee.h"

#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// The solve's threadgroup, as mnee.metal's grid writes it.
constexpr NS::UInteger solve_group = 64;

}  // namespace

MneePass::MneePass(const Library& library)
    : grid_(library.compute_pipeline("mnee_grid")),
      solve_(library.compute_pipeline("mnee_solve")),
      fold_(library.compute_pipeline("mnee_fold")) {
    if (solve_->maxTotalThreadsPerThreadgroup() < solve_group) {
        throw MetalError("MneePass: the solve's pipeline takes fewer threads a group than its grid's " +
                         std::to_string(solve_group));
    }
}

void MneePass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources, Into into) const {
    const bool folds = into == Into::accumulated_image;
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.mnee_requests == 0 ||
        resources.mnee_header == 0 || resources.non_finite_counter == 0 ||
        (folds ? resources.mnee_light == 0 || resources.accumulation == nullptr : resources.light == 0)) {
        throw MetalError("MneePass: the frame lacks one of its scene, transforms, glows, MNEE work list, counter of "
                         "samples not finite, or the light it adds to");
    }
    namespace binding = bindings::mnee;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.non_finite_counter, binding::non_finite);
    arguments->setAddress(resources.mnee_requests, binding::requests);
    arguments->setAddress(resources.mnee_header, binding::header);
    arguments->setAddress(resources.mnee_grid, binding::grid);
    arguments->setAddress(folds ? resources.mnee_light : resources.light, binding::light);
    if (folds) {
        arguments->setTexture(resources.accumulation->gpuResourceID(), binding::accumulated);
    }
    const auto barrier = [&] {
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };

    barrier();
    encoder->setComputePipelineState(grid_.get());
    encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
    barrier();
    encoder->setComputePipelineState(solve_.get());
    encoder->dispatchThreads(resources.mnee_grid);
    if (folds) {
        barrier();
        encoder->setComputePipelineState(fold_.get());
        dispatch_per_pixel(encoder, fold_.get(), resources.size);
    }
}

}  // namespace serenity::metal
