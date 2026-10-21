// ReSTIR PT's paired spatial reuse's programs (restir_pt_spatial.h): the
// shifts of steps 8 and 10 of passes/restir_pt/restir_pt.cuh, Metal's
// restir_pt_shift's and restir_pt_replay's (metal/passes/restir_pt_spatial/
// restir_pt_spatial.metal), as ray generation programs reading the frame's
// launch parameters (frame/traced_module.cuh), each one thread a pair of
// one of the pairing's two lists (cuda/passes/restir_pt/shift_replay.h).

#include <optix.h>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "cuda/frame/traced_module.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/passes/restir_pt_spatial/partner.cuh"

namespace serenity {
namespace shaders {

// Steps 8 and 10: the shift of one pair of a list (the pairing's,
// restir_pt_spatial_kernels.cu), one thread a place, launched at the
// list's capacity, a thread past its count returning: this pixel's path
// shifted to its partner, into the pair's slot.
__device__ inline void spatial_shift_item(const uint* items,
                                          const uint* counter) {
  const uint n = optixGetLaunchIndex().x;
  if (n >= *counter) {
    return;
  }
  const restir_pt::Traced traced =
      restir_pt::traced(launch.frame, launch.path_settings);
  const uint item = items[n];
  const uint i = restir_pt::replay_pixel(item);
  const uint t = restir_pt::replay_slot(item);
  const uint2 pixel = traced.slot_pixel(i);
  const spatial::Partner partner =
      spatial::partner_of(launch.pairing, pixel, t, traced);
  const restir_pt::Scene scene{&launch.structures, launch.scene,
                               launch.transforms, launch.glows,
                               launch.light_power};
  launch.shifts[i * serenity::contracts::max_path_neighbors + t] =
      restir_pt::stored_shift(
          shift::shift_path(scene,
                            shift::unpack_primary_hit(
                                launch.primaries[traced.slot(partner.pixel)]),
                            launch.path_reservoirs[i], launch.path_settings));
}

}  // namespace shaders
}  // namespace serenity

// Step 8: the pairs that reconnect at x2.
extern "C" __global__ void __raygen__restir_pt_reconnect() {
  serenity::shaders::spatial_shift_item(launch.reconnects,
                                        launch.reconnect_counter);
}

// Step 10: the pairs that replay.
extern "C" __global__ void __raygen__restir_pt_replay() {
  serenity::shaders::spatial_shift_item(launch.replays, launch.replay_counter);
}
