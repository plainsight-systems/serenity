// ReSTIR PT's temporal reuse's programs (restir_pt_temporal.h): steps 5 and
// 6 of passes/restir_pt/restir_pt.cuh, Metal's restir_pt_temporal_shift and
// restir_pt_temporal_replay (metal/passes/restir_pt_temporal/
// restir_pt_temporal.metal) as ray generation programs reading the frame's
// launch parameters (frame/traced_module.cuh): the shift one thread a
// traced pixel (restir_pt.cuh, Traced), the replay one thread a place of
// the replay list, launched at its capacity, a thread past the list's count
// returning (no launch takes its size from the GPU).

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

// Step 5: each pixel's temporal input found; its two shifts done here
// where they reconnect at x2, appended to the replay list where they
// replay, a warp's at once (device/append.cuh). Every thread reaches the
// warp's reserve: none returns before it.
__device__ inline void temporal_shift() {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const serenity::contracts::PathReuseConstants& reuse = launch.path_constants;
  const uint3 index = optixGetLaunchIndex();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(make_uint2(index.x, index.y));
  const bool inside = traced.inside(pixel);
  const uint i = traced.slot(pixel);
  uint appended[2];
  uint count = 0u;
  if (inside) {
    const serenity::contracts::PrimaryHit own_hit = launch.primaries[i];
    uint j = restir_pt::no_pixel;
    if (reuse.previous != 0u && own_hit.distance > 0.0f) {
      j = restir_pt::previous_pixel(
          reuse.previous_camera, to_float3(launch.previous_positions[i]),
          decode_unit(own_hit.normal), launch.last_primaries, traced,
          restir_pt::ring_start(frame.frame_index));
    }
    launch.previous_pixels[i] = j;
    if (j != restir_pt::no_pixel) {
      const restir_pt::Scene now{&launch.structures, launch.scene,
                                 launch.transforms, launch.glows,
                                 launch.light_power};
      const restir_pt::Scene then{&launch.structures, launch.scene,
                                  launch.last_transforms, launch.last_glows,
                                  launch.last_light_power};
      const serenity::contracts::PathReservoir last =
          launch.last_path_finals[j];
      const serenity::contracts::PathReservoir own = launch.path_reservoirs[i];
      float4* const shifts = launch.shifts;
      if (!restir_pt::holds_path(last)) {
        shifts[i * temporal::slots + temporal::last_here] = make_float4(0.0f);
      } else if (restir_pt::replays(last)) {
        appended[count++] = restir_pt::replay_item(i, temporal::last_here);
      } else {
        shifts[i * temporal::slots + temporal::last_here] =
            restir_pt::stored_shift(shift::shift_path(
                now, shift::unpack_primary_hit(own_hit), last, settings));
      }
      if (!restir_pt::holds_path(own)) {
        shifts[i * temporal::slots + temporal::own_there] = make_float4(0.0f);
      } else if (restir_pt::replays(own)) {
        appended[count++] = restir_pt::replay_item(i, temporal::own_there);
      } else {
        shifts[i * temporal::slots + temporal::own_there] =
            restir_pt::stored_shift(shift::shift_path(
                then, shift::unpack_primary_hit(launch.last_primaries[j]), own,
                settings));
      }
    }
  }
  // The warp's appended pairs, one atomic for all (GDSA.10).
  const uint first = warp_reserve(launch.replay_counter, count);
  for (uint n = 0u; n < count; ++n) {
    launch.replays[first + n] = appended[n];
  }
}

// Step 6: the shifts that replay, one thread each. Which frame's scene,
// which primary hit and which path are data, chosen per lane, so a warp
// runs one shift whatever its lanes' slots (GPU.4).
__device__ inline void temporal_replay() {
  const uint n = optixGetLaunchIndex().x;
  if (n >= *launch.replay_counter) {
    return;
  }
  const uint item = launch.replays[n];
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

extern "C" __global__ void __raygen__restir_pt_temporal_shift() {
  serenity::shaders::temporal_shift();
}

extern "C" __global__ void __raygen__restir_pt_temporal_replay() {
  serenity::shaders::temporal_replay();
}
