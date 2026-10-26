// ReSTIR PT's paired spatial reuse (restir_pt_spatial.h): steps 8 to 11 of
// metal/passes/restir_pt/restir_pt.metal.h, the sequence of
// restir_pt/shift_replay.h: the pairing, the lists' grids (for both reuses),
// the shift over each list, the merge; one thread a traced pixel where a
// kernel runs one a pixel (restir_pt.metal.h, Traced).

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

// The paired neighbor of the pixel `pixel` in texture `t` this frame, and
// whether it is in the image; with the checkerboard, over the traced
// field's lattice (metal/reuse/pairing.metal.h, lattice_partner()).
struct Partner {
  bool inside;
  uint2 pixel;
};

Partner partner_of(constant ushort* pairing, uint2 pixel, uint t,
                   restir_pt::Traced traced) {
  const pairing::Framing map = pairing::framing(traced.frame, t);
  const int2 p = traced.checkerboard
                     ? pairing::lattice_partner(pairing, int2(pixel), t, map,
                                                int(traced.frame & 1u))
                     : pairing::partner(pairing, int2(pixel), t, map);
  const bool inside = p.x >= 0 && p.y >= 0 && p.x < int(traced.width) &&
                      p.y < int(traced.height);
  return Partner{inside, inside ? uint2(p) : uint2(0u)};
}

}  // namespace

// Step 8's pairing, one thread a traced pixel: each of its partners not
// paired (outside the image, or not similar) or with nothing to shift
// stored as such; each pair that shifts appended to the list of its kind,
// the pairs that reconnect at x2 or those that replay (restir_pt.metal.h,
// replays()), a SIMD group's at once (GDSA.10). Every thread reaches both
// reserves: none returns before them.
kernel void restir_pt_pair(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    const device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    const device serenity::contracts::PrimaryHit* primaries
    [[buffer(binding::primaries)]],
    constant ushort* pairing [[buffer(binding::pairing)]],
    device float4* shifts [[buffer(binding::shifts)]],
    device uint* reconnects [[buffer(binding::reconnects)]],
    device uint* replays [[buffer(binding::pairs)]],
    device atomic_uint* counts [[buffer(binding::counts)]],
    uint2 grid_place [[thread_position_in_grid]]) {
  namespace c = serenity::contracts;
  constexpr uint slots = c::max_path_neighbors;
  // the run's, the pipeline's constants (reuse_constants.h)
  const c::PathReuseSettings settings = reuse_settings();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  const bool inside = traced.inside(pixel);
  const uint i = traced.slot(pixel);
  uint reconnecting[slots];
  uint replaying[slots];
  uint reconnect_count = 0u;
  uint replay_count = 0u;
  if (inside) {
    const c::PathReservoir reservoir = reservoirs[i];
    const bool held = c::path_end(reservoir.flags) != c::PathEnd::none;
    const bool replay = restir_pt::replays(reservoir);
    const c::PrimaryHit own = primaries[i];
    for (uint t = 0u; t < settings.neighbors; ++t) {
      const Partner partner = partner_of(pairing, pixel, t, traced);
      const uint j = traced.slot(partner.pixel);
      if (!partner.inside || !restir_pt::similar(own, primaries[j])) {
        shifts[i * slots + t] = float4(0.0f, 0.0f, 0.0f, restir_pt::not_paired);
      } else if (!held) {
        shifts[i * slots + t] = float4(0.0f);
      } else if (replay) {
        replaying[replay_count++] = restir_pt::replay_item(i, t);
      } else {
        reconnecting[reconnect_count++] = restir_pt::replay_item(i, t);
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

// Steps 6 and 9, one thread: both lists' lengths as their grids, and both
// counts set back to 0 for the next lists (the spatial reuse's, after the
// temporal's, or the next frame's), which a barrier orders after the
// shifts that read the grids.
kernel void restir_pt_pair_grids(device atomic_uint* counts
                                 [[buffer(binding::counts)]],
                                 device serenity::gpu::IndirectGrid* grids
                                 [[buffer(binding::dispatch)]]) {
  namespace c = serenity::contracts;
  for (uint list = 0u; list < c::pair_lists; ++list) {
    const uint count =
        atomic_load_explicit(counts + list, memory_order_relaxed);
    atomic_store_explicit(counts + list, 0u, memory_order_relaxed);
    grids[list] = indirect_grid(count, c::replay_group);
  }
}

// Steps 8 and 10: the shift of one pair of a list, one thread a place,
// whichever list (the host dispatches it over each in turn): this pixel's
// path shifted to its partner, into the pair's slot.
kernel void restir_pt_spatial_shift(
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
    constant ushort* pairing [[buffer(binding::pairing)]],
    device float4* shifts [[buffer(binding::shifts)]],
    const device uint* items [[buffer(binding::pairs)]],
    const device serenity::gpu::IndirectGrid* grid
    [[buffer(binding::dispatch)]],
    const device float* light_power [[buffer(binding::light_power)]],
    uint n [[thread_position_in_grid]]) {
  // the run's, the pipeline's constants (reuse_constants.h)
  const serenity::contracts::PathReuseSettings settings = reuse_settings();
  if (n >= grid->count) {
    return;
  }
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint item = items[n];
  const uint i = restir_pt::replay_pixel(item);
  const uint t = restir_pt::replay_slot(item);
  const uint2 pixel = traced.slot_pixel(i);
  const Partner partner = partner_of(pairing, pixel, t, traced);
  const restir_pt::Scene scene{structure, &block, transforms, sphere_glows,
                               light_power};
  shifts[i * serenity::contracts::max_path_neighbors + t] =
      restir_pt::stored_shift(shift::shift_path(
          scene,
          shift::unpack_primary_hit(primaries[traced.slot(partner.pixel)]),
          reservoirs[i], settings));
}

// Step 11: the merge, one thread a pixel; its estimate added to the light
// buffer (its own pixel's alone, so in place); with temporal reuse in the
// graph, the path it keeps written as the frame's final reservoir.
kernel void restir_pt_spatial(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    const device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    constant ushort* pairing [[buffer(binding::pairing)]],
    const device float4* shifts [[buffer(binding::shifts)]],
    device float4* light [[buffer(binding::light)]],
    device serenity::contracts::PathReservoir* finals
    [[buffer(binding::finals)]],
    uint2 grid_place [[thread_position_in_grid]]) {
  // the run's, the pipeline's constants (reuse_constants.h)
  const serenity::contracts::PathReuseSettings settings = reuse_settings();
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  if (!traced.inside(pixel)) {
    return;
  }
  constexpr uint slots = serenity::contracts::max_path_neighbors;
  const uint i = traced.slot(pixel);
  const serenity::contracts::PathReservoir own = reservoirs[i];
  // k: the partners similar enough to have been tried (path_merge.metal.h).
  uint k = 0u;
  for (uint t = 0u; t < settings.neighbors; ++t) {
    k += shifts[i * slots + t].w != restir_pt::not_paired ? 1u : 0u;
  }
  const float own_target = luminance(to_float3(own.integrand));
  const float own_count = float(serenity::contracts::path_count(own.flags));
  // The path kept for the next frame (path_merge.metal.h): picked by the
  // scalar weights, k + 1 times theirs, which the shared factor leaves
  // the pick's odds alone; only the graph with temporal reuse reads it.
  const bool keeps = settings.confidence_cap > 0.0f;
  PathNumbers numbers = purpose_numbers(path_numbers(pixel, frame.frame_index),
                                        restir_pt::spatial_pick_first);
  float weight_sum = 0.0f;
  float count_sum = own_count;
  serenity::contracts::PathReservoir kept = own;
  float canonical = 1.0f;
  float3 sum = float3(0.0f);
  for (uint t = 0u; t < settings.neighbors && k > 0u; ++t) {
    // this pixel's path at its partner
    const float4 outward = shifts[i * slots + t];
    if (outward.w == restir_pt::not_paired) {
      continue;
    }
    const Partner partner = partner_of(pairing, pixel, t, traced);
    const uint j = traced.slot(partner.pixel);
    const serenity::contracts::PathReservoir theirs = reservoirs[j];
    const float their_count =
        float(serenity::contracts::path_count(theirs.flags));
    count_sum += their_count;
    canonical += path_merge::canonical_share(own_target, own_count,
                                             luminance(outward.xyz) * outward.w,
                                             their_count, k);
    const float4 inward = shifts[j * slots + t];  // the partner's path here
    if (inward.w > 0.0f) {
      const float m = path_merge::neighbor_weight(
          luminance(to_float3(theirs.integrand)), their_count,
          luminance(inward.xyz) * inward.w, own_count, k);
      sum += m * inward.xyz * inward.w * theirs.weight;
      if (keeps &&
          stream_pick(weight_sum,
                      m * luminance(inward.xyz) * inward.w * theirs.weight,
                      next_number(numbers))) {
        kept = path_merge::shifted_reservoir(theirs, inward.xyz, inward.w);
      }
    }
  }
  sum += canonical * restir_pt::estimate(own);
  light[i] += float4(sum / float(k + 1u), 0.0f);
  if (!keeps) {
    return;
  }
  // The canonical path last, its own odds the same in any order.
  if (stream_pick(weight_sum, canonical * own_target * own.weight,
                  next_number(numbers))) {
    kept = own;
  }
  const float target = luminance(to_float3(kept.integrand));
  kept.weight = target > 0.0f && weight_sum > 0.0f
                    ? weight_sum / (float(k + 1u) * target)
                    : 0.0f;
  kept.flags = path_merge::with_count(kept.flags, count_sum);
  finals[i] = kept;
}
