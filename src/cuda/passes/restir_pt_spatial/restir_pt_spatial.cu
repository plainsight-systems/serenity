// ReSTIR PT's paired spatial reuse's programs (restir_pt_spatial.h): steps
// 8 and 10 of passes/restir_pt/restir_pt.cuh, Metal's restir_pt_shift and
// restir_pt_replay (metal/passes/restir_pt_spatial/restir_pt_spatial.metal)
// as ray generation programs reading the frame's launch parameters
// (frame/traced_module.cuh): the shift one thread a traced pixel
// (restir_pt.cuh, Traced), the replay one thread a place of the replay
// list, launched at its capacity, a thread past the list's count returning.

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

// Step 8: each pixel's path shifted to each similar partner where it
// reconnects at x2; the pairs that replay appended to the list, a warp's at
// once (device/append.cuh). Every thread reaches the warp's reserve: none
// returns before it.
__device__ inline void spatial_shift() {
  constexpr uint slots = serenity::contracts::max_path_neighbors;
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const uint3 index = optixGetLaunchIndex();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(make_uint2(index.x, index.y));
  const bool inside = traced.inside(pixel);
  const uint i = traced.slot(pixel);
  uint appended[slots];
  uint count = 0u;
  if (inside) {
    const restir_pt::Scene scene{&launch.structures, launch.scene,
                                 launch.transforms, launch.glows,
                                 launch.light_power};
    const serenity::contracts::PathReservoir reservoir =
        launch.path_reservoirs[i];
    const bool held = serenity::contracts::path_end(reservoir.flags) !=
                      serenity::contracts::PathEnd::none;
    const serenity::contracts::PrimaryHit own = launch.primaries[i];
    for (uint t = 0u; t < settings.neighbors; ++t) {
      const spatial::Partner partner =
          spatial::partner_of(launch.pairing, pixel, t, traced);
      const uint j = traced.slot(partner.pixel);
      if (!partner.inside || !restir_pt::similar(own, launch.primaries[j])) {
        launch.shifts[i * slots + t] =
            make_float4(0.0f, 0.0f, 0.0f, restir_pt::not_paired);
      } else if (!held) {
        launch.shifts[i * slots + t] = make_float4(0.0f);
      } else if (restir_pt::replays(reservoir)) {
        appended[count++] = restir_pt::replay_item(i, t);
      } else {
        launch.shifts[i * slots + t] =
            restir_pt::stored_shift(shift::shift_path(
                scene, shift::unpack_primary_hit(launch.primaries[j]),
                reservoir, settings));
      }
    }
  }
  // The warp's appended pairs, one atomic for all (GDSA.10).
  const uint first = warp_reserve(launch.replay_counter, count);
  for (uint n = 0u; n < count; ++n) {
    launch.replays[first + n] = appended[n];
  }
}

// Step 10: the pairs that replay, one thread each.
__device__ inline void spatial_replay() {
  const uint n = optixGetLaunchIndex().x;
  if (n >= *launch.replay_counter) {
    return;
  }
  const restir_pt::Traced traced =
      restir_pt::traced(launch.frame, launch.path_settings);
  const uint item = launch.replays[n];
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

extern "C" __global__ void __raygen__restir_pt_shift() {
  serenity::shaders::spatial_shift();
}

extern "C" __global__ void __raygen__restir_pt_replay() {
  serenity::shaders::spatial_replay();
}
