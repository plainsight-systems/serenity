#include "metal/passes/restir_pt_initial/restir_pt_initial.h"

#include <algorithm>

#include "core/contracts/light_tile.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

RestirPtInitialPass::RestirPtInitialPass(const Library& library, bool mnee)
    : tiles_(library.compute_pipeline("restir_pt_light_tiles")),
      pipeline_(library.compute_pipeline(mnee ? "restir_pt_initial_mnee" : "restir_pt_initial")) {
    if (mnee) {
        mnee_.emplace(library);
    }
}

void RestirPtInitialPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.scene == 0 || resources.transforms == 0 || resources.glows == 0 || resources.camera == 0 ||
        resources.path_settings == 0 || resources.path_reservoirs == 0 || resources.primaries == 0 ||
        resources.light == 0 || resources.light_tiles == 0) {
        throw MetalError("RestirPtInitialPass: the frame lacks one of its scene, transforms, glows, camera, ReSTIR "
                         "PT's settings, path reservoirs, primary hits, light buffer or light tiles");
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
    arguments->setAddress(resources.light_tiles, binding::light_tiles);
    // Unread without temporal reuse, where they are 0: bound then to arrays
    // that exist (the light buffer, this frame's transforms), since Metal's
    // validation refuses a null binding the kernel declares.
    arguments->setAddress(resources.previous_positions != 0 ? resources.previous_positions : resources.light,
                          binding::previous_positions);
    arguments->setAddress(resources.last_transforms != 0 ? resources.last_transforms : resources.transforms,
                          binding::last_transforms);
    if (mnee_) {
        // MNEE's work list, the mnee kernel's alone.
        if (resources.mnee_requests == 0 || resources.mnee_header == 0) {
            throw MetalError("RestirPtInitialPass: the frame lacks MNEE's work list");
        }
        arguments->setAddress(resources.mnee_requests, binding::mnee_requests);
        arguments->setAddress(resources.mnee_header, binding::mnee_header);
    }
    encoder->setComputePipelineState(tiles_.get());
    encoder->dispatchThreads(MTL::Size{contracts::light_tiles * contracts::light_tile_size, 1, 1},
                             MTL::Size{std::min<NS::UInteger>(tiles_->maxTotalThreadsPerThreadgroup(), 256), 1, 1});
    // The tiles, which the tree's candidates read (GPU.8: a write and its reader).
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
    if (mnee_) {
        // The tree's MNEE requests solved, their light into the light buffer.
        mnee_->record(encoder, resources, MneePass::Into::light_buffer);
    }
}

}  // namespace serenity::metal
