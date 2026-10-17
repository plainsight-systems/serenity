#include "metal/passes/restir_pt_initial/restir_pt_initial.h"

#include <algorithm>

#include "core/contracts/light_tile.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

namespace {

// The light tiles' threadgroup, a thread an entry, as many as the pipeline
// takes up to this: the host's choice alone, which the kernel, reading
// only its grid position, does not assume.
constexpr NS::UInteger tiles_group = 256;

}  // namespace

RestirPtInitialPass::RestirPtInitialPass(const Library& library)
    : tiles_(library.compute_pipeline("restir_pt_light_tiles")),
      pipeline_(library.compute_pipeline("restir_pt_initial")),
      mnee_(library) {}

void RestirPtInitialPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.path_settings == 0 || resources.path_reservoirs == 0 || resources.primaries == 0 ||
        resources.light == 0 || resources.light_tiles == 0 || resources.mnee_requests == 0 ||
        resources.mnee_header == 0) {
        throw MetalError("RestirPtInitialPass: the frame lacks one of its ReSTIR PT's settings, path reservoirs, "
                         "primary hits, light buffer, light tiles or MNEE's work list");
    }
    namespace binding = bindings::restir_pt;
    MTL4::ArgumentTable* arguments = resources.arguments;
    bind_scene(resources, SceneCamera::bound, "RestirPtInitialPass");
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
    arguments->setAddress(resources.mnee_requests, binding::mnee_requests);
    arguments->setAddress(resources.mnee_header, binding::mnee_header);
    encoder->setComputePipelineState(tiles_.get());
    encoder->dispatchThreads(
        MTL::Size{contracts::light_tiles * contracts::light_tile_size, 1, 1},
        MTL::Size{std::min<NS::UInteger>(tiles_->maxTotalThreadsPerThreadgroup(), tiles_group), 1, 1});
    // The tiles, which the tree's candidates read (GPU.8: a write and its reader).
    barrier_after_dispatch(encoder);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.traced);  // a thread a traced pixel
    // The tree's MNEE requests solved, their light into the light buffer.
    mnee_.record(encoder, resources);
}

}  // namespace serenity::metal
