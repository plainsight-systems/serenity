// ReSTIR PT's temporal reuse's kernels (restir_pt_temporal.h), which trace
// nothing, taking the frame's launch parameters by address
// (frame/frame_launch.h): the pairing, step 5's search and choices before
// its shifts (cuda/passes/restir_pt/shift_replay.h), and step 7, the Talbot
// merge, Metal's restir_pt_temporal (metal/passes/restir_pt_temporal/
// restir_pt_temporal.metal).

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "cuda/device/append.cuh"
#include "cuda/frame/frame_launch.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/passes/restir_pt_temporal/temporal_slots.cuh"

namespace serenity {
namespace shaders {

// Step 5's pairing, one thread a traced pixel: its temporal input found
// (the pixel last frame, written for the shifts and the merge); each of its
// two shifts with no path to shift stored as such, each that shifts
// appended to the list of its kind, the pairs that reconnect at x2 or those
// that replay (restir_pt.cuh, replays()), a warp's at once
// (device/append.cuh). Every thread reaches both reserves: none returns
// before them.
__device__ inline void temporal_pair(
    const serenity::cuda::FrameLaunch& launch) {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const serenity::contracts::PathReuseConstants& reuse = launch.path_constants;
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel =
      traced.pixel(make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y));
  const bool inside = traced.inside(pixel);
  const uint i = traced.slot(pixel);
  uint reconnecting[2];
  uint replaying[2];
  uint reconnects = 0u;
  uint replays = 0u;
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
      // Each shift's path: last frame's, shifted here, and this pixel's,
      // shifted there.
      const serenity::contracts::PathReservoir paths[2] = {
          launch.last_path_finals[j], launch.path_reservoirs[i]};
      const uint path_slots[2] = {temporal::last_here, temporal::own_there};
      for (uint s = 0u; s < 2u; ++s) {
        const uint slot = path_slots[s];
        if (!restir_pt::holds_path(paths[s])) {
          launch.shifts[i * temporal::slots + slot] = make_float4(0.0f);
        } else if (restir_pt::replays(paths[s])) {
          replaying[replays++] = restir_pt::replay_item(i, slot);
        } else {
          reconnecting[reconnects++] = restir_pt::replay_item(i, slot);
        }
      }
    }
  }
  // The warp's appended pairs, one atomic a list for all (GDSA.10).
  const uint first_reconnect =
      warp_reserve(launch.reconnect_counter, reconnects);
  for (uint n = 0u; n < reconnects; ++n) {
    launch.reconnects[first_reconnect + n] = reconnecting[n];
  }
  const uint first_replay = warp_reserve(launch.replay_counter, replays);
  for (uint n = 0u; n < replays; ++n) {
    launch.replays[first_replay + n] = replaying[n];
  }
}

// Step 7: the Talbot merge, one thread a traced pixel, into the reservoirs
// in place where a spatial reuse follows, else into the frame's final ones,
// its estimate then added to the light buffer (its own pixel's alone).
__device__ inline void temporal_merge(
    const serenity::cuda::FrameLaunch& launch) {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel =
      traced.pixel(make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y));
  if (!traced.inside(pixel)) {
    return;
  }
  const uint i = traced.slot(pixel);
  const serenity::contracts::PathReservoir own = launch.path_reservoirs[i];
  const uint j = launch.previous_pixels[i];
  restir_pt::Merged merged{own, restir_pt::estimate(own)};
  if (j != restir_pt::no_pixel) {
    // The cap at last's pixel, lowered by last frame's duplication map in a
    // graph that has one (cap_exponent 0 otherwise, its map unread).
    const float cap = settings.cap_exponent > 0.0f
                          ? path_merge::duplication_cap(
                                settings.confidence_cap, settings.cap_floor,
                                settings.cap_exponent, launch.duplication[j])
                          : settings.confidence_cap;
    PathNumbers numbers = purpose_numbers(
        path_numbers(pixel, frame.frame_index), restir_pt::temporal_pick_first);
    merged = restir_pt::temporal_merge(
        own, launch.last_path_finals[j],
        launch.shifts[i * temporal::slots + temporal::last_here],
        launch.shifts[i * temporal::slots + temporal::own_there], cap,
        next_numbers2(numbers));
  }
  // neighbors is 0 exactly in a graph without restir_pt_spatial
  // (contracts/path_reuse_settings.h).
  if (settings.neighbors != 0u) {
    launch.path_reservoirs[i] = merged.reservoir;
    return;
  }
  launch.path_finals[i] = merged.reservoir;
  launch.light[i] += make_float4(merged.estimate, 0.0f);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_temporal_pair(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::temporal_pair(*launch);
}

extern "C" __global__ void restir_pt_temporal(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::temporal_merge(*launch);
}
