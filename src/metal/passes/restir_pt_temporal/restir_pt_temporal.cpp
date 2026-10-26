#include "metal/passes/restir_pt_temporal/restir_pt_temporal.h"

#include "metal/device/error.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

RestirPtTemporalPass::RestirPtTemporalPass(
    const Library& library, const contracts::PathReuseSettings& settings)
    : reuse_(library, settings,
             {.pair = "restir_pt_temporal_pair",
              .shift = "restir_pt_temporal_shift",
              .merge = "restir_pt_temporal"},
             "RestirPtTemporalPass") {}

void RestirPtTemporalPass::record(MTL4::ComputeCommandEncoder* encoder,
                                  const FrameResources& resources) const {
  if (resources.path_constants == 0 || resources.path_reservoirs == 0 ||
      resources.primaries == 0 || resources.last_primaries == 0 ||
      resources.light == 0 || resources.shifts == 0 ||
      resources.path_finals == 0 || resources.last_path_finals == 0 ||
      resources.previous_positions == 0 || resources.previous_pixels == 0 ||
      resources.duplication == 0 || resources.last_transforms == 0 ||
      resources.last_glows == 0 || resources.light_power == 0 ||
      resources.last_light_power == 0) {
    throw MetalError(
        "RestirPtTemporalPass: the frame lacks one of its ReSTIR PT's reuse "
        "constants, "
        "path reservoirs, either frame's primary hits or final reservoirs, "
        "light buffer, shifted "
        "paths, previous positions or "
        "pixels, duplication map, "
        "or last frame's transforms, glows or power pyramid");
  }
  namespace binding = bindings::restir_pt;
  MTL4::ArgumentTable* arguments = resources.arguments;
  bind_scene(resources, SceneCamera::none, "RestirPtTemporalPass");
  arguments->setAddress(resources.path_reservoirs, binding::reservoirs);
  arguments->setAddress(resources.primaries, binding::primaries);
  arguments->setAddress(resources.light, binding::light);
  arguments->setAddress(resources.shifts, binding::shifts);
  arguments->setAddress(resources.path_constants, binding::path_constants);
  arguments->setAddress(resources.last_primaries, binding::last_primaries);
  arguments->setAddress(resources.path_finals, binding::finals);
  arguments->setAddress(resources.last_path_finals, binding::last_finals);
  arguments->setAddress(resources.previous_positions,
                        binding::previous_positions);
  arguments->setAddress(resources.previous_pixels, binding::previous_pixels);
  arguments->setAddress(resources.duplication, binding::duplication);
  arguments->setAddress(resources.last_transforms, binding::last_transforms);
  arguments->setAddress(resources.last_glows, binding::last_glows);
  arguments->setAddress(resources.light_power, binding::light_power);
  arguments->setAddress(resources.last_light_power, binding::last_light_power);
  reuse_.record(encoder, resources);  // binds its lists
}

}  // namespace serenity::metal
