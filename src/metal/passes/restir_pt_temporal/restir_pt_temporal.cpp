#include "metal/passes/restir_pt_temporal/restir_pt_temporal.h"

#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// The replay's threadgroup, as restir_pt_spatial.metal's grid writes it
// (restir_pt.metal.h, replay_group).
constexpr NS::UInteger replay_group = 64;

}  // namespace

RestirPtTemporalPass::RestirPtTemporalPass(const Library& library)
    : shift_(library.compute_pipeline("restir_pt_temporal_shift")),
      grid_(library.compute_pipeline("restir_pt_replay_grid")),
      replay_(library.compute_pipeline("restir_pt_temporal_replay")),
      merge_(library.compute_pipeline("restir_pt_temporal")) {
    if (replay_->maxTotalThreadsPerThreadgroup() < replay_group) {
        throw MetalError("RestirPtTemporalPass: the replay's pipeline takes fewer threads a group than its grid's " +
                         std::to_string(replay_group));
    }
}

void RestirPtTemporalPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.path_settings == 0 ||
        resources.path_constants == 0 || resources.path_reservoirs == 0 || resources.primaries == 0 ||
        resources.last_primaries == 0 || resources.light == 0 || resources.shifts == 0 || resources.replays == 0 ||
        resources.replay_counter == 0 || resources.replay_grid == 0 || resources.path_finals == 0 ||
        resources.last_path_finals == 0 || resources.previous_positions == 0 || resources.previous_pixels == 0 ||
        resources.duplication == 0 || resources.last_transforms == 0 || resources.last_glows == 0) {
        throw MetalError("RestirPtTemporalPass: the frame lacks one of its scene, transforms, glows, ReSTIR PT's "
                         "settings and reuse constants, path reservoirs, either frame's primary hits or final "
                         "reservoirs, light buffer, shifted paths, replaying shifts, their count or grid, previous "
                         "positions or pixels, duplication map, or last frame's transforms or glows");
    }
    namespace binding = bindings::restir_pt;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setResource(resources.acceleration, binding::structure);
    arguments->setAddress(resources.scene, binding::scene);
    arguments->setAddress(resources.glows, binding::glows);
    arguments->setAddress(resources.transforms, binding::transforms);
    arguments->setAddress(resources.path_settings, binding::settings);
    arguments->setAddress(resources.path_reservoirs, binding::reservoirs);
    arguments->setAddress(resources.primaries, binding::primaries);
    arguments->setAddress(resources.light, binding::light);
    arguments->setAddress(resources.shifts, binding::shifts);
    arguments->setAddress(resources.replays, binding::replays);
    arguments->setAddress(resources.replay_counter, binding::counter);
    arguments->setAddress(resources.replay_grid, binding::dispatch);
    arguments->setAddress(resources.path_constants, binding::path_constants);
    arguments->setAddress(resources.last_primaries, binding::last_primaries);
    arguments->setAddress(resources.path_finals, binding::finals);
    arguments->setAddress(resources.last_path_finals, binding::last_finals);
    arguments->setAddress(resources.previous_positions, binding::previous_positions);
    arguments->setAddress(resources.previous_pixels, binding::previous_pixels);
    arguments->setAddress(resources.duplication, binding::duplication);
    arguments->setAddress(resources.last_transforms, binding::last_transforms);
    arguments->setAddress(resources.last_glows, binding::last_glows);
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
