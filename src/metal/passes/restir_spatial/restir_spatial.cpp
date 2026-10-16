#include "metal/passes/restir_spatial/restir_spatial.h"

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

RestirSpatialPass::RestirSpatialPass(const Library& library) : pipeline_(library.compute_pipeline("restir_spatial")) {}

void RestirSpatialPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.reuse == 0 || resources.records == 0 || resources.working_reservoirs == 0 ||
        resources.final_reservoirs == 0) {
        throw MetalError("RestirSpatialPass: the frame lacks one of its reuse constants, surface records, working "
                         "reservoirs or final reservoirs");
    }
    // Bindings: passes/bindings.h, as restir_spatial.metal declares them.
    namespace binding = bindings::restir_spatial;
    MTL4::ArgumentTable* arguments = resources.arguments;
    bind_scene(resources, SceneCamera::bound, "RestirSpatialPass");
    arguments->setAddress(resources.reuse, binding::reuse);
    arguments->setAddress(resources.records, binding::records);
    arguments->setAddress(resources.working_reservoirs, binding::input);
    arguments->setAddress(resources.final_reservoirs, binding::output);
    encoder->setComputePipelineState(pipeline_.get());
    dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
