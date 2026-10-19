#include "metal/passes/restir_pt_spatial/restir_pt_spatial.h"

#include "metal/device/error.h"
#include "metal/passes/bindings.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

RestirPtSpatialPass::RestirPtSpatialPass(const Library& library)
    : reuse_(library, "restir_pt_shift", "restir_pt_replay",
             "restir_pt_spatial", "RestirPtSpatialPass") {}

void RestirPtSpatialPass::record(MTL4::ComputeCommandEncoder* encoder,
                                 const FrameResources& resources) const {
  if (resources.path_settings == 0 || resources.pairing == 0 ||
      resources.path_reservoirs == 0 || resources.primaries == 0 ||
      resources.light == 0 || resources.shifts == 0 || resources.replays == 0 ||
      resources.replay_counter == 0 || resources.replay_grid == 0 ||
      resources.light_power == 0) {
    throw MetalError(
        "RestirPtSpatialPass: the frame lacks one of its ReSTIR PT's settings, "
        "reuse textures, path "
        "reservoirs, primary hits, light buffer, shifted paths, replaying "
        "pairs, their count or their "
        "grid, or the power pyramid");
  }
  namespace binding = bindings::restir_pt;
  MTL4::ArgumentTable* arguments = resources.arguments;
  bind_scene(resources, SceneCamera::bound, "RestirPtSpatialPass");
  arguments->setAddress(resources.path_settings, binding::settings);
  arguments->setAddress(resources.path_reservoirs, binding::reservoirs);
  arguments->setAddress(resources.primaries, binding::primaries);
  arguments->setAddress(resources.light, binding::light);
  arguments->setAddress(resources.pairing, binding::pairing);
  arguments->setAddress(resources.shifts, binding::shifts);
  arguments->setAddress(resources.replays, binding::replays);
  arguments->setAddress(resources.replay_counter, binding::counter);
  arguments->setAddress(resources.replay_grid, binding::dispatch);
  arguments->setAddress(resources.light_power, binding::light_power);
  // Unwritten without temporal reuse, where it is 0: bound then to the
  // reservoirs, since Metal's validation refuses a null binding.
  arguments->setAddress(resources.path_finals != 0 ? resources.path_finals
                                                   : resources.path_reservoirs,
                        binding::finals);
  reuse_.record(encoder, resources.traced, resources.replay_grid);
}

}  // namespace serenity::metal
