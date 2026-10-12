// ReSTIR PT's paired spatial reuse (restir_pt_spatial.h): steps 5 to 8 of
// metal/passes/restir_pt/restir_pt.metal.h, four kernels in order, each
// behind a barrier.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
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

// Step 5: each pixel's path shifted to each similar partner where it
// reconnects at x2; the pairs that replay appended to the list, a SIMD
// group's at once (restir_pt.metal.h). Every thread reaches the group's
// scan: none returns before it.
kernel void restir_pt_shift(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                            constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                            primitive_acceleration_structure structure [[buffer(binding::structure)]],
                            constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                            constant float* sphere_glows [[buffer(binding::glows)]],
                            constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                            constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                            const device serenity::contracts::PathReservoir* reservoirs
                            [[buffer(binding::reservoirs)]],
                            const device uint2* primaries [[buffer(binding::primaries)]],
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
        const uint2 own = primaries[i];
        for (uint t = 0u; t < settings.neighbors; ++t) {
            const Partner partner = partner_of(pairing, pixel, t, frame.frame_index, size);
            if (!partner.inside || !restir_pt::similar(own, primaries[partner.pixel.y * size.x + partner.pixel.x])) {
                shifts[i * serenity::contracts::max_path_neighbors + t] = float4(0.0f, 0.0f, 0.0f, restir_pt::not_paired);
            } else if (!held) {
                shifts[i * serenity::contracts::max_path_neighbors + t] = float4(0.0f);
            } else if (restir_pt::replays(reservoir)) {
                appended[count++] = i * 4u + t;
            } else {
                shifts[i * serenity::contracts::max_path_neighbors + t] = restir_pt::stored_shift(
                    shift::shift_path(scene, restir_pt::primary_ray(camera, partner.pixel, frame.frame_index, size),
                                      reservoir, settings));
            }
        }
    }
    // The group's appended pairs, one atomic for all (GDSA.10).
    const uint before = simd_prefix_exclusive_sum(count);
    const uint total = simd_sum(count);
    uint base = 0u;
    if (simd_is_first() && total > 0u) {
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
    }
    base = simd_broadcast_first(base);
    for (uint n = 0u; n < count; ++n) {
        replays[base + before + n] = appended[n];
    }
}

// The replay's indirect grid, as Metal reads MTLDispatchThreadsIndirectArguments:
// the grid's size, then the threadgroup's; and the count it was made from.
struct ReplayGrid {
    uint grid[3];
    uint group[3];
    uint count;
};

constant constexpr uint replay_group = 64u;

// Step 6, one thread: the list's length as the replay's grid, and the count
// set back to 0 for the next frame's step 5, which the frame barrier orders
// after this frame's step 7 reads the grid.
kernel void restir_pt_replay_grid(device atomic_uint* counter [[buffer(binding::counter)]],
                                  device ReplayGrid* grid [[buffer(binding::dispatch)]]) {
    const uint count = atomic_load_explicit(counter, memory_order_relaxed);
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
    // At least one thread: a grid of none is not one Metal takes.
    *grid = ReplayGrid{{max(count, 1u), 1u, 1u}, {replay_group, 1u, 1u}, count};
}

// Step 7: the pairs that replay, one thread each.
kernel void restir_pt_replay(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                             constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                             primitive_acceleration_structure structure [[buffer(binding::structure)]],
                             constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                             constant float* sphere_glows [[buffer(binding::glows)]],
                             constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                             constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                             const device serenity::contracts::PathReservoir* reservoirs
                             [[buffer(binding::reservoirs)]],
                             constant ushort* pairing [[buffer(binding::pairing)]],
                             device float4* shifts [[buffer(binding::shifts)]],
                             const device uint* replays [[buffer(binding::replays)]],
                             const device ReplayGrid* grid [[buffer(binding::dispatch)]],
                             uint n [[thread_position_in_grid]]) {
    if (n >= grid->count) {
        return;
    }
    const uint2 size = uint2(frame.width, frame.height);
    const uint item = replays[n];
    const uint i = item / 4u;
    const uint t = item % 4u;
    const uint2 pixel = uint2(i % size.x, i / size.x);
    const Partner partner = partner_of(pairing, pixel, t, frame.frame_index, size);
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    shifts[i * serenity::contracts::max_path_neighbors + t] = restir_pt::stored_shift(shift::shift_path(
        scene, restir_pt::primary_ray(camera, partner.pixel, frame.frame_index, size), reservoirs[i], settings));
}

// Step 8: the merge, one thread a pixel; its estimate added to the light
// buffer (its own pixel's alone, so in place).
kernel void restir_pt_spatial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                              constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                              const device serenity::contracts::PathReservoir* reservoirs
                              [[buffer(binding::reservoirs)]],
                              constant ushort* pairing [[buffer(binding::pairing)]],
                              const device float4* shifts [[buffer(binding::shifts)]],
                              device float4* light [[buffer(binding::light)]],
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
        canonical += path_merge::canonical_share(own_target, own_count, luminance(outward.xyz) * outward.w,
                                                 their_count, k);
        const float4 inward = shifts[j * slots + t];  // the partner's path here
        if (inward.w > 0.0f) {
            const float m = path_merge::neighbor_weight(luminance(to_float3(theirs.integrand)), their_count,
                                                        luminance(inward.xyz) * inward.w, own_count, k);
            sum += m * inward.xyz * inward.w * theirs.weight;
        }
    }
    sum += canonical * restir_pt::estimate(own);
    light[i] += float4(sum / float(k + 1u), 0.0f);
}
