// ReSTIR PT's temporal reuse (restir_pt_temporal.h): steps 5 to 7 of
// metal/passes/restir_pt/restir_pt.metal.h, the sequence of
// restir_pt/shift_replay.h: the pairing, the lists' grids (the spatial
// pass's kernel, restir_pt_spatial.metal's restir_pt_pair_grids), the shift
// over each list, the merge; one thread a traced pixel where a kernel runs
// one a pixel (restir_pt.metal.h, Traced).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/transform.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"
#include "metal/passes/restir_pt/reuse_constants.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

namespace {

constant constexpr uint slots = serenity::contracts::max_path_neighbors;

// The temporal shifts' two slots of a pixel's shifted paths (step 5): last
// frame's path here, and this pixel's path there, last frame.
constant constexpr uint last_here = 0u;
constant constexpr uint own_there = 1u;
static_assert(own_there < restir_pt::replay_slots,
              "the temporal shifts fit a pixel's replay slots");

}  // namespace

// Step 5's pairing, one thread a traced pixel: its temporal input found
// (the pixel last frame, written for the shifts and the merge); each of its
// two shifts with no path to shift stored as such, each that shifts
// appended to the list of its kind, the pairs that reconnect at x2 or those
// that replay (restir_pt.metal.h, replays()), a SIMD group's at once
// (GDSA.10). Every thread reaches both reserves: none returns before them.
kernel void restir_pt_temporal_pair(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    const device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    const device serenity::contracts::PrimaryHit* primaries
    [[buffer(binding::primaries)]],
    device float4* shifts [[buffer(binding::shifts)]],
    device uint* reconnects [[buffer(binding::reconnects)]],
    device uint* replays [[buffer(binding::pairs)]],
    device atomic_uint* counts [[buffer(binding::counts)]],
    constant serenity::contracts::PathReuseConstants& reuse
    [[buffer(binding::path_constants)]],
    const device serenity::contracts::PrimaryHit* last_primaries
    [[buffer(binding::last_primaries)]],
    const device serenity::contracts::PathReservoir* last_finals
    [[buffer(binding::last_finals)]],
    const device serenity::contracts::Float3* previous_positions
    [[buffer(binding::previous_positions)]],
    device uint* previous_pixels [[buffer(binding::previous_pixels)]],
    uint2 grid_place [[thread_position_in_grid]]) {
  namespace c = serenity::contracts;
  // the run's, the pipeline's constants (reuse_constants.h)
  const c::PathReuseSettings settings = reuse_settings();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  const bool inside = traced.inside(pixel);
  const uint i = traced.slot(pixel);
  uint reconnecting[2];
  uint replaying[2];
  uint reconnect_count = 0u;
  uint replay_count = 0u;
  if (inside) {
    const c::PrimaryHit own_hit = primaries[i];
    uint j = restir_pt::no_pixel;
    if (reuse.previous != 0u && own_hit.distance > 0.0f) {
      j = restir_pt::previous_pixel(
          reuse.previous_camera, to_float3(previous_positions[i]),
          decode_unit(own_hit.normal), last_primaries, traced,
          restir_pt::ring_start(frame.frame_index));
    }
    previous_pixels[i] = j;
    if (j != restir_pt::no_pixel) {
      // Each shift's path: last frame's, shifted here, and this pixel's,
      // shifted there.
      const c::PathReservoir paths[2] = {last_finals[j], reservoirs[i]};
      const uint path_slots[2] = {last_here, own_there};
      for (uint k = 0u; k < 2u; ++k) {
        const uint slot = path_slots[k];
        if (!restir_pt::holds_path(paths[k])) {
          shifts[i * slots + slot] = float4(0.0f);
        } else if (restir_pt::replays(paths[k])) {
          replaying[replay_count++] = restir_pt::replay_item(i, slot);
        } else {
          reconnecting[reconnect_count++] = restir_pt::replay_item(i, slot);
        }
      }
    }
  }
  // The group's appended pairs, one atomic a list for all (GDSA.10).
  const uint first_reconnect =
      simd_reserve(counts + c::reconnect_list, reconnect_count);
  for (uint n = 0u; n < reconnect_count; ++n) {
    reconnects[first_reconnect + n] = reconnecting[n];
  }
  const uint first_replay = simd_reserve(counts + c::replay_list, replay_count);
  for (uint n = 0u; n < replay_count; ++n) {
    replays[first_replay + n] = replaying[n];
  }
}

// Steps 5 and 6: the shift of one pair of a list, one thread a place,
// whichever list (the host dispatches it over each in turn). Which frame's
// scene, which primary hit and which path are data, chosen per lane by the
// pair's slot, so a SIMD group runs one shift whatever its lanes' slots
// (GPU.4).
kernel void restir_pt_temporal_shift(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms
    [[buffer(binding::transforms)]],
    const device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    const device serenity::contracts::PrimaryHit* primaries
    [[buffer(binding::primaries)]],
    device float4* shifts [[buffer(binding::shifts)]],
    const device uint* items [[buffer(binding::pairs)]],
    const device serenity::gpu::IndirectGrid* grid
    [[buffer(binding::dispatch)]],
    const device serenity::contracts::PrimaryHit* last_primaries
    [[buffer(binding::last_primaries)]],
    const device serenity::contracts::PathReservoir* last_finals
    [[buffer(binding::last_finals)]],
    const device uint* previous_pixels [[buffer(binding::previous_pixels)]],
    constant serenity::contracts::Transform* last_transforms
    [[buffer(binding::last_transforms)]],
    constant float* last_glows [[buffer(binding::last_glows)]],
    const device float* light_power [[buffer(binding::light_power)]],
    const device float* last_light_power [[buffer(binding::last_light_power)]],
    uint n [[thread_position_in_grid]]) {
  // the run's, the pipeline's constants (reuse_constants.h)
  const serenity::contracts::PathReuseSettings settings = reuse_settings();
  if (n >= grid->count) {
    return;
  }
  const uint item = items[n];
  const uint i = restir_pt::replay_pixel(item);
  const uint slot = restir_pt::replay_slot(item);
  const uint j = previous_pixels[i];
  const bool here = slot == last_here;
  const restir_pt::Scene scene{
      structure, &block, here ? transforms : last_transforms,
      here ? sphere_glows : last_glows, here ? light_power : last_light_power};
  shifts[i * slots + slot] = restir_pt::stored_shift(shift::shift_path(
      scene, shift::unpack_primary_hit(here ? primaries[i] : last_primaries[j]),
      here ? last_finals[j] : reservoirs[i], settings));
}

// Step 7: the Talbot merge, one thread a pixel, into the reservoirs in
// place where a spatial reuse follows, else into the frame's final ones,
// its estimate then added to the light buffer (its own pixel's alone).
kernel void restir_pt_temporal(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    device float4* light [[buffer(binding::light)]],
    const device float4* shifts [[buffer(binding::shifts)]],
    device serenity::contracts::PathReservoir* finals
    [[buffer(binding::finals)]],
    const device serenity::contracts::PathReservoir* last_finals
    [[buffer(binding::last_finals)]],
    const device uint* previous_pixels [[buffer(binding::previous_pixels)]],
    const device float* duplication [[buffer(binding::duplication)]],
    uint2 grid_place [[thread_position_in_grid]]) {
  // the run's, the pipeline's constants (reuse_constants.h)
  const serenity::contracts::PathReuseSettings settings = reuse_settings();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  if (!traced.inside(pixel)) {
    return;
  }
  const uint i = traced.slot(pixel);
  const serenity::contracts::PathReservoir own = reservoirs[i];
  const uint j = previous_pixels[i];
  restir_pt::Merged merged{own, restir_pt::estimate(own)};
  if (j != restir_pt::no_pixel) {
    // The cap at last's pixel, lowered by last frame's duplication map
    // in a graph that has one (cap_exponent 0 otherwise, its map unread).
    const float cap = settings.cap_exponent > 0.0f
                          ? path_merge::duplication_cap(
                                settings.confidence_cap, settings.cap_floor,
                                settings.cap_exponent, duplication[j])
                          : settings.confidence_cap;
    PathNumbers numbers = purpose_numbers(
        path_numbers(pixel, frame.frame_index), restir_pt::temporal_pick_first);
    merged = restir_pt::temporal_merge(
        own, last_finals[j], shifts[i * slots + last_here],
        shifts[i * slots + own_there], cap, next_numbers2(numbers));
  }
  // neighbors is 0 exactly in a graph without restir_pt_spatial
  // (contracts/path_reuse_settings.h).
  if (settings.neighbors != 0u) {
    reservoirs[i] = merged.reservoir;
    return;
  }
  finals[i] = merged.reservoir;
  light[i] += float4(merged.estimate, 0.0f);
}
