// ReSTIR PT's temporal reuse's programs (restir_pt_temporal.h): the shifts
// of steps 5 and 6 of passes/restir_pt/restir_pt.cuh, Metal's
// restir_pt_temporal_shift's and restir_pt_temporal_replay's
// (metal/passes/restir_pt_temporal/restir_pt_temporal.metal), as ray
// generation programs reading the frame's launch parameters
// (frame/traced_module.cuh), each one thread a pair of one of the pairing's
// two lists (cuda/passes/restir_pt/shift_replay.h).

#include <optix.h>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/transform.h"
#include "cuda/frame/traced_module.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/passes/restir_pt_temporal/temporal_slots.cuh"

namespace serenity {
namespace shaders {

// Steps 5 and 6: the shift of one pair of a list (the pairing's,
// restir_pt_temporal_kernels.cu), one thread a place, launched at the
// list's capacity, a thread past its count returning. Which frame's scene,
// which primary hit and which path are data, chosen per lane by the
// pair's slot, so a warp runs one shift whatever its lanes' slots (GPU.4).
__device__ inline void temporal_shift_item(const uint* items,
                                           const uint* counter) {
  const uint n = optixGetLaunchIndex().x;
  if (n >= *counter) {
    return;
  }
  const uint item = items[n];
  const uint i = restir_pt::replay_pixel(item);
  const uint slot = restir_pt::replay_slot(item);
  const uint j = launch.previous_pixels[i];
  const bool here = slot == temporal::last_here;
  const restir_pt::Scene scene{
      &launch.structures, launch.scene,
      here ? launch.transforms : launch.last_transforms,
      here ? launch.glows : launch.last_glows,
      here ? static_cast<const float*>(launch.light_power)
           : launch.last_light_power};
  launch.shifts[i * temporal::slots + slot] =
      restir_pt::stored_shift(shift::shift_path(
          scene,
          shift::unpack_primary_hit(here ? launch.primaries[i]
                                         : launch.last_primaries[j]),
          here ? launch.last_path_finals[j] : launch.path_reservoirs[i],
          launch.path_settings));
}

}  // namespace shaders
}  // namespace serenity

// Step 5: the pairs that reconnect at x2.
extern "C" __global__ void __raygen__restir_pt_temporal_reconnect() {
  serenity::shaders::temporal_shift_item(launch.reconnects,
                                         launch.reconnect_counter);
}

// Step 6: the pairs that replay.
extern "C" __global__ void __raygen__restir_pt_temporal_replay() {
  serenity::shaders::temporal_shift_item(launch.replays, launch.replay_counter);
}
