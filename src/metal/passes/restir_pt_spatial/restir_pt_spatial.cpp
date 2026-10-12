#include "metal/passes/restir_pt_spatial/restir_pt_spatial.h"

#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// The replay's threadgroup, as restir_pt_spatial.metal's grid writes it.
constexpr NS::UInteger replay_group = 64;

}  // namespace

RestirPtSpatialPass::RestirPtSpatialPass(const Library& library)
    : shift_(library.compute_pipeline("restir_pt_shift")),
      grid_(library.compute_pipeline("restir_pt_replay_grid")),
      replay_(library.compute_pipeline("restir_pt_replay")),
      merge_(library.compute_pipeline("restir_pt_spatial")) {
    if (replay_->maxTotalThreadsPerThreadgroup() < replay_group) {
        throw MetalError("RestirPtSpatialPass: the replay's pipeline takes fewer threads a group than its grid's " +
                         std::to_string(replay_group));
    }
}

void RestirPtSpatialPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.path_settings == 0 || resources.pairing == 0 || resources.path_reservoirs == 0 ||
        resources.primaries == 0 || resources.light == 0 || resources.shifts == 0 || resources.replays == 0 ||
        resources.replay_counter == 0 || resources.replay_grid == 0) {
        throw MetalError("RestirPtSpatialPass: the frame lacks one of its scene, transforms, glows, camera, ReSTIR "
                         "PT's settings, reuse textures, path reservoirs, primary hits, light buffer, shifted paths, "
                         "replaying pairs, their count or their grid");
    }
    namespace binding = bindings::restir_pt;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.camera, binding::camera);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.path_settings, binding::settings);
    arguments->setAddress(resources.path_reservoirs, binding::reservoirs);
    arguments->setAddress(resources.primaries, binding::primaries);
    arguments->setAddress(resources.light, binding::light);
    arguments->setAddress(resources.pairing, binding::pairing);
    arguments->setAddress(resources.shifts, binding::shifts);
    arguments->setAddress(resources.replays, binding::replays);
    arguments->setAddress(resources.replay_counter, binding::counter);
    arguments->setAddress(resources.replay_grid, binding::dispatch);
    const auto barrier = [&] {
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };

    encoder->setComputePipelineState(shift_.get());
    dispatch_per_pixel(encoder, shift_.get(), resources.size);
    barrier();
    encoder->setComputePipelineState(grid_.get());
    encoder->dispatchThreads(MTL::Size{1, 1, 1}, MTL::Size{1, 1, 1});
    barrier();
    encoder->setComputePipelineState(replay_.get());
    encoder->dispatchThreads(resources.replay_grid);
    barrier();
    encoder->setComputePipelineState(merge_.get());
    dispatch_per_pixel(encoder, merge_.get(), resources.size);
}

}  // namespace serenity::metal
