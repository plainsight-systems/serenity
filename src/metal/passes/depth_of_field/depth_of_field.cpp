#include "metal/passes/depth_of_field/depth_of_field.h"

#include <cstdint>

#include "core/contracts/depth_of_field.h"
#include "core/frame/extent.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/frame/depth_of_field_images.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

DepthOfFieldPass::DepthOfFieldPass(const Library& library)
    : presort_(library.compute_pipeline("dof_presort")),
      tiles_(library.compute_pipeline("dof_tiles")),
      dilate_(library.compute_pipeline("dof_dilate")),
      gather_(library.compute_pipeline("dof_gather")),
      median_(library.compute_pipeline("dof_median")),
      composite_(library.compute_pipeline("dof_composite")) {}

void DepthOfFieldPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    const DepthOfFieldImages* images = resources.depth_of_field;
    if (resources.radiance == nullptr || resources.full_depth == nullptr ||
        resources.depth_of_field_constants == 0 || images == nullptr || images->presort() == nullptr) {
        throw MetalError("DepthOfFieldPass: the frame lacks the radiance image, the full-size depth, the pass's "
                         "constants or its images");
    }
    // Bindings: passes/bindings.h, as depth_of_field.metal declares them;
    // one layout for every kernel, each reading what it names.
    namespace binding = bindings::depth_of_field;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.depth_of_field_constants, binding::constants);
    arguments->setAddress(images->tiles(), binding::tiles);
    arguments->setAddress(images->dilated(), binding::dilated);
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::radiance);
    arguments->setTexture(resources.full_depth->gpuResourceID(), binding::depth);
    arguments->setTexture(images->presort()->gpuResourceID(), binding::presort);
    arguments->setTexture(images->far()->gpuResourceID(), binding::far);
    arguments->setTexture(images->near()->gpuResourceID(), binding::near);
    arguments->setTexture(images->far_filtered()->gpuResourceID(), binding::far_filtered);
    arguments->setTexture(images->near_filtered()->gpuResourceID(), binding::near_filtered);

    // The grids (contract 21): the output's, half of it, and its tiles.
    const frame::Extent output = resources.size;
    const frame::Extent half{static_cast<std::uint32_t>(images->presort()->width()),
                             static_cast<std::uint32_t>(images->presort()->height())};
    const frame::Extent tiles{(half.width + contracts::dof_tile - 1u) / contracts::dof_tile,
                              (half.height + contracts::dof_tile - 1u) / contracts::dof_tile};
    // Each step after a barrier on the one before (depth_of_field.h; GPU.8).
    const auto step = [&](const NS::SharedPtr<MTL::ComputePipelineState>& pipeline, frame::Extent grid, bool first) {
        if (!first) {
            barrier_after_dispatch(encoder);
        }
        encoder->setComputePipelineState(pipeline.get());
        dispatch_per_pixel(encoder, pipeline.get(), grid);
    };
    step(presort_, half, true);
    step(tiles_, tiles, false);
    step(dilate_, tiles, false);
    step(gather_, half, false);
    step(median_, half, false);
    step(composite_, output, false);
}

}  // namespace serenity::metal
