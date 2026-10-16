// ReSTIR PT's temporal reuse (restir_pt_temporal.h): steps 5 to 7 of
// metal/passes/restir_pt/restir_pt.metal.h, three kernels and the spatial
// pass's replay grid (restir_pt_spatial.metal, restir_pt_replay_grid) in
// order, each behind a barrier.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/transform.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"

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
static_assert(own_there < restir_pt::replay_slots, "the temporal shifts fit a pixel's replay slots");

}  // namespace

// Step 5: each pixel's temporal input found; its two shifts done here
// where they reconnect at x2, appended to the replay list where they
// replay, a SIMD group's at once (as restir_pt_shift). Every thread reaches
// the group's scan: none returns before it.
kernel void restir_pt_temporal_shift(
    constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
    primitive_acceleration_structure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
    constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
    const device serenity::contracts::PathReservoir* reservoirs [[buffer(binding::reservoirs)]],
    const device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
    device float4* shifts [[buffer(binding::shifts)]], device uint* replays [[buffer(binding::replays)]],
    device atomic_uint* counter [[buffer(binding::counter)]],
    constant serenity::contracts::PathReuseConstants& reuse [[buffer(binding::path_constants)]],
    const device serenity::contracts::PrimaryHit* last_primaries [[buffer(binding::last_primaries)]],
    const device serenity::contracts::PathReservoir* last_finals [[buffer(binding::last_finals)]],
    const device serenity::contracts::Float3* previous_positions [[buffer(binding::previous_positions)]],
    device uint* previous_pixels [[buffer(binding::previous_pixels)]],
    constant serenity::contracts::Transform* last_transforms [[buffer(binding::last_transforms)]],
    constant float* last_glows [[buffer(binding::last_glows)]], uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(frame.width, frame.height);
    const bool inside = pixel.x < size.x && pixel.y < size.y;
    const uint i = pixel.y * size.x + pixel.x;
    uint appended[2];
    uint count = 0u;
    if (inside) {
        const serenity::contracts::PrimaryHit own_hit = primaries[i];
        uint j = restir_pt::no_pixel;
        if (reuse.previous != 0u && own_hit.distance > 0.0f) {
            j = restir_pt::previous_pixel(reuse.previous_camera, to_float3(previous_positions[i]),
                                          decode_unit(own_hit.normal), last_primaries, size,
                                          restir_pt::ring_start(frame.frame_index));
        }
        previous_pixels[i] = j;
        if (j != restir_pt::no_pixel) {
            const restir_pt::Scene now{structure, &block, transforms, sphere_glows};
            const restir_pt::Scene then{structure, &block, last_transforms, last_glows};
            const serenity::contracts::PathReservoir last = last_finals[j];
            const serenity::contracts::PathReservoir own = reservoirs[i];
            if (!restir_pt::holds_path(last)) {
                shifts[i * slots + last_here] = float4(0.0f);
            } else if (restir_pt::replays(last)) {
                appended[count++] = restir_pt::replay_item(i, last_here);
            } else {
                shifts[i * slots + last_here] = restir_pt::stored_shift(
                    shift::shift_path(now, shift::unpack_primary_hit(own_hit), last, settings));
            }
            if (!restir_pt::holds_path(own)) {
                shifts[i * slots + own_there] = float4(0.0f);
            } else if (restir_pt::replays(own)) {
                appended[count++] = restir_pt::replay_item(i, own_there);
            } else {
                shifts[i * slots + own_there] = restir_pt::stored_shift(
                    shift::shift_path(then, shift::unpack_primary_hit(last_primaries[j]), own, settings));
            }
        }
    }
    // The group's appended pairs, one atomic for all (GDSA.10).
    const uint first = simd_reserve(counter, count);
    for (uint n = 0u; n < count; ++n) {
        replays[first + n] = appended[n];
    }
}

// Step 6: the shifts that replay, one thread each. Which frame's scene, which
// primary hit and which path are data, chosen per lane, so a SIMD group runs
// one shift whatever its lanes' slots (GPU.4).
kernel void restir_pt_temporal_replay(
    constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
    primitive_acceleration_structure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
    constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
    const device serenity::contracts::PathReservoir* reservoirs [[buffer(binding::reservoirs)]],
    const device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
    device float4* shifts [[buffer(binding::shifts)]], const device uint* replays [[buffer(binding::replays)]],
    const device serenity::gpu::IndirectGrid* grid [[buffer(binding::dispatch)]],
    const device serenity::contracts::PrimaryHit* last_primaries [[buffer(binding::last_primaries)]],
    const device serenity::contracts::PathReservoir* last_finals [[buffer(binding::last_finals)]],
    const device uint* previous_pixels [[buffer(binding::previous_pixels)]],
    constant serenity::contracts::Transform* last_transforms [[buffer(binding::last_transforms)]],
    constant float* last_glows [[buffer(binding::last_glows)]], uint n [[thread_position_in_grid]]) {
    if (n >= grid->count) {
        return;
    }
    const uint item = replays[n];
    const uint i = restir_pt::replay_pixel(item);
    const uint slot = restir_pt::replay_slot(item);
    const uint j = previous_pixels[i];
    const bool here = slot == last_here;
    const restir_pt::Scene scene{structure, &block, here ? transforms : last_transforms,
                                 here ? sphere_glows : last_glows};
    shifts[i * slots + slot] = restir_pt::stored_shift(
        shift::shift_path(scene, shift::unpack_primary_hit(here ? primaries[i] : last_primaries[j]),
                          here ? last_finals[j] : reservoirs[i], settings));
}

// Step 7: the Talbot merge, one thread a pixel, into the reservoirs in
// place where a spatial reuse follows, else into the frame's final ones,
// its estimate then added to the light buffer (its own pixel's alone).
kernel void restir_pt_temporal(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                               constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                               device serenity::contracts::PathReservoir* reservoirs [[buffer(binding::reservoirs)]],
                               device float4* light [[buffer(binding::light)]],
                               const device float4* shifts [[buffer(binding::shifts)]],
                               device serenity::contracts::PathReservoir* finals [[buffer(binding::finals)]],
                               const device serenity::contracts::PathReservoir* last_finals
                               [[buffer(binding::last_finals)]],
                               const device uint* previous_pixels [[buffer(binding::previous_pixels)]],
                               const device float* duplication [[buffer(binding::duplication)]],
                               uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(frame.width, frame.height);
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    const uint i = pixel.y * size.x + pixel.x;
    const serenity::contracts::PathReservoir own = reservoirs[i];
    const uint j = previous_pixels[i];
    restir_pt::Merged merged{own, restir_pt::estimate(own)};
    if (j != restir_pt::no_pixel) {
        // The cap at last's pixel, lowered by last frame's duplication map
        // in a graph that has one (cap_exponent 0 otherwise, its map unread).
        const float cap =
            settings.cap_exponent > 0.0f
                ? path_merge::duplication_cap(settings.confidence_cap, settings.cap_floor, settings.cap_exponent,
                                              duplication[j])
                : settings.confidence_cap;
        PathNumbers numbers = purpose_numbers(path_numbers(pixel, frame.frame_index), restir_pt::temporal_pick_first);
        merged = restir_pt::temporal_merge(own, last_finals[j], shifts[i * slots + last_here],
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
