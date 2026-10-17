// ReSTIR PT's paired spatial reuse (restir_pt_spatial.h): steps 8 to 11 of
// metal/passes/restir_pt/restir_pt.metal.h, four kernels in order, each
// behind a barrier.

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

// The paired neighbor of the pixel `pixel` in texture `t` this frame, and
// whether it is in the image.
struct Partner {
    bool inside;
    uint2 pixel;
};

Partner partner_of(constant ushort* pairing, uint2 pixel, uint t, uint frame, uint2 size) {
    const int2 p = pairing::partner(pairing, int2(pixel), t, pairing::framing(frame, t));
    const bool inside = p.x >= 0 && p.y >= 0 && p.x < int(size.x) && p.y < int(size.y);
    return Partner{inside, inside ? uint2(p) : uint2(0u)};
}

}  // namespace

// Step 8: each pixel's path shifted to each similar partner where it
// reconnects at x2; the pairs that replay appended to the list, a SIMD
// group's at once (restir_pt.metal.h). Every thread reaches the group's
// scan: none returns before it.
kernel void restir_pt_shift(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                            serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                            constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                            constant float* sphere_glows [[buffer(binding::glows)]],
                            constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                            constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                            const device serenity::contracts::PathReservoir* reservoirs
                            [[buffer(binding::reservoirs)]],
                            const device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
                            constant ushort* pairing [[buffer(binding::pairing)]],
                            device float4* shifts [[buffer(binding::shifts)]],
                            device uint* replays [[buffer(binding::replays)]],
                            device atomic_uint* counter [[buffer(binding::counter)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(frame.width, frame.height);
    const bool inside = pixel.x < size.x && pixel.y < size.y;
    const uint i = pixel.y * size.x + pixel.x;
    uint appended[serenity::contracts::max_path_neighbors];
    uint count = 0u;
    if (inside) {
        const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
        const serenity::contracts::PathReservoir reservoir = reservoirs[i];
        const bool held = serenity::contracts::path_end(reservoir.flags) != serenity::contracts::PathEnd::none;
        const serenity::contracts::PrimaryHit own = primaries[i];
        for (uint t = 0u; t < settings.neighbors; ++t) {
            const Partner partner = partner_of(pairing, pixel, t, frame.frame_index, size);
            const uint j = partner.pixel.y * size.x + partner.pixel.x;
            if (!partner.inside || !restir_pt::similar(own, primaries[j])) {
                shifts[i * serenity::contracts::max_path_neighbors + t] =
                    float4(0.0f, 0.0f, 0.0f, restir_pt::not_paired);
            } else if (!held) {
                shifts[i * serenity::contracts::max_path_neighbors + t] = float4(0.0f);
            } else if (restir_pt::replays(reservoir)) {
                appended[count++] = restir_pt::replay_item(i, t);
            } else {
                shifts[i * serenity::contracts::max_path_neighbors + t] = restir_pt::stored_shift(
                    shift::shift_path(scene, shift::unpack_primary_hit(primaries[j]), reservoir, settings));
            }
        }
    }
    // The group's appended pairs, one atomic for all (GDSA.10).
    const uint first = simd_reserve(counter, count);
    for (uint n = 0u; n < count; ++n) {
        replays[first + n] = appended[n];
    }
}

// Steps 6 and 9, one thread: the list's length as the replay's grid, and
// the count set back to 0 for the next list (the spatial reuse's, after
// the temporal's, or the next frame's), which a barrier orders after the
// replay that reads the grid.
kernel void restir_pt_replay_grid(device atomic_uint* counter [[buffer(binding::counter)]],
                                  device serenity::gpu::IndirectGrid* grid [[buffer(binding::dispatch)]]) {
    const uint count = atomic_load_explicit(counter, memory_order_relaxed);
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
    *grid = indirect_grid(count, serenity::contracts::replay_group);
}

// Step 10: the pairs that replay, one thread each.
kernel void restir_pt_replay(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                             serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                             constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                             constant float* sphere_glows [[buffer(binding::glows)]],
                             constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                             constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                             const device serenity::contracts::PathReservoir* reservoirs
                             [[buffer(binding::reservoirs)]],
                             const device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
                             constant ushort* pairing [[buffer(binding::pairing)]],
                             device float4* shifts [[buffer(binding::shifts)]],
                             const device uint* replays [[buffer(binding::replays)]],
                             const device serenity::gpu::IndirectGrid* grid [[buffer(binding::dispatch)]],
                             uint n [[thread_position_in_grid]]) {
    if (n >= grid->count) {
        return;
    }
    const uint2 size = uint2(frame.width, frame.height);
    const uint item = replays[n];
    const uint i = restir_pt::replay_pixel(item);
    const uint t = restir_pt::replay_slot(item);
    const uint2 pixel = uint2(i % size.x, i / size.x);
    const Partner partner = partner_of(pairing, pixel, t, frame.frame_index, size);
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    shifts[i * serenity::contracts::max_path_neighbors + t] = restir_pt::stored_shift(shift::shift_path(
        scene, shift::unpack_primary_hit(primaries[partner.pixel.y * size.x + partner.pixel.x]), reservoirs[i],
        settings));
}

// Step 11: the merge, one thread a pixel; its estimate added to the light
// buffer (its own pixel's alone, so in place); with temporal reuse in the
// graph, the path it keeps written as the frame's final reservoir.
kernel void restir_pt_spatial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                              constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                              const device serenity::contracts::PathReservoir* reservoirs
                              [[buffer(binding::reservoirs)]],
                              constant ushort* pairing [[buffer(binding::pairing)]],
                              const device float4* shifts [[buffer(binding::shifts)]],
                              device float4* light [[buffer(binding::light)]],
                              device serenity::contracts::PathReservoir* finals [[buffer(binding::finals)]],
                              uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(frame.width, frame.height);
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    constexpr uint slots = serenity::contracts::max_path_neighbors;
    const uint i = pixel.y * size.x + pixel.x;
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
    PathNumbers numbers = purpose_numbers(path_numbers(pixel, frame.frame_index), restir_pt::spatial_pick_first);
    float weight_sum = 0.0f;
    float count_sum = own_count;
    serenity::contracts::PathReservoir kept = own;
    float canonical = 1.0f;
    float3 sum = float3(0.0f);
    for (uint t = 0u; t < settings.neighbors && k > 0u; ++t) {
        const float4 outward = shifts[i * slots + t];  // this pixel's path at its partner
        if (outward.w == restir_pt::not_paired) {
            continue;
        }
        const Partner partner = partner_of(pairing, pixel, t, frame.frame_index, size);
        const uint j = partner.pixel.y * size.x + partner.pixel.x;
        const serenity::contracts::PathReservoir theirs = reservoirs[j];
        const float their_count = float(serenity::contracts::path_count(theirs.flags));
        count_sum += their_count;
        canonical += path_merge::canonical_share(own_target, own_count, luminance(outward.xyz) * outward.w,
                                                 their_count, k);
        const float4 inward = shifts[j * slots + t];  // the partner's path here
        if (inward.w > 0.0f) {
            const float m = path_merge::neighbor_weight(luminance(to_float3(theirs.integrand)), their_count,
                                                        luminance(inward.xyz) * inward.w, own_count, k);
            sum += m * inward.xyz * inward.w * theirs.weight;
            if (keeps && stream_pick(weight_sum, m * luminance(inward.xyz) * inward.w * theirs.weight,
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
    if (stream_pick(weight_sum, canonical * own_target * own.weight, next_number(numbers))) {
        kept = own;
    }
    const float target = luminance(to_float3(kept.integrand));
    kept.weight = target > 0.0f && weight_sum > 0.0f ? weight_sum / (float(k + 1u) * target) : 0.0f;
    kept.flags = path_merge::with_count(kept.flags, count_sum);
    finals[i] = kept;
}
