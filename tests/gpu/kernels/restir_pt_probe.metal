// ReSTIR PT's pieces on their own (tests/gpu/restir_pt_test.cpp): the
// reconnection criteria on given pairs, the paired neighbors of a frame,
// a path tree's path shifted to another pixel, the temporal merge and the
// duplication map, each through the functions the passes call. Each
// kernel's comment lists its bindings in order (gpu/support/
// probe_runner.h), its records in restir_pt_probe.h.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/camera.h"
#include "core/contracts/checkerboard.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/transform.h"
#include "metal/integrator/reconnection.metal.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"
#include "metal/reuse/duplication.metal.h"
#include "metal/reuse/pairing.metal.h"
#include "metal/reuse/path_merge.metal.h"
#include "restir_pt_probe.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
using serenity::tests::CriteriaQuery;
using serenity::tests::CriteriaResult;
using serenity::tests::PathMergeQuery;
using serenity::tests::PathMergeResult;
using serenity::tests::PairingQuery;
using serenity::tests::PairingResult;
using serenity::tests::ShiftQuery;
using serenity::tests::ShiftResult;

// The reconnection criteria on given pairs. Bindings: the results at 0, the
// queries at 1, their count at 2.
kernel void pt_criteria(device CriteriaResult* out [[buffer(0)]], const device CriteriaQuery* queries [[buffer(1)]],
                        constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const CriteriaQuery q = queries[i];
    serenity::contracts::Bsdf before{};
    before.kind = (q.lobe_before & serenity::contracts::lobe_glossy) != 0u ? serenity::contracts::BsdfKind::conductor
                                                                          : serenity::contracts::BsdfKind::lambert;
    before.alpha = q.alpha_before;
    const float threshold = reconnection::footprint_threshold(q.footprint, to_float3(q.camera), to_float3(q.primary),
                                                              to_float3(q.primary_normal));
    const reconnection::Pair pair{
        reconnection::lobe_roughness(q.lobe_before, before),
        reconnection::footprint((q.lobe_before & serenity::contracts::lobe_delta) != 0u, q.density_before,
                                to_float3(q.before), to_float3(q.at), to_float3(q.at_normal)),
        reconnection::inverse_skipped(q.lobe_at)
            ? reconnection::largest
            : reconnection::footprint((q.lobe_at & serenity::contracts::lobe_delta) != 0u, q.density_at,
                                      to_float3(q.at), to_float3(q.before), to_float3(q.before_normal))};
    out[i] = CriteriaResult{reconnection::connectable(pair, threshold, q.min_roughness) ? 1u : 0u, threshold,
                            pair.forward, pair.inverse};
}

// Each pixel's partner in a frame's pairing, and its partner's. Bindings:
// the results at 0, the query (frame, texture, image size) at 1, the reuse
// textures at 2.
kernel void pt_pairing(device PairingResult* out [[buffer(0)]], constant PairingQuery& q [[buffer(1)]],
                       constant ushort* textures [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= q.width * q.height) {
        return;
    }
    const pairing::Framing map = pairing::framing(q.frame, q.texture);
    const int2 pixel = int2(int(i % q.width), int(i / q.width));
    const int2 partner = pairing::partner(textures, pixel, q.texture, map);
    const int2 back = pairing::partner(textures, partner, q.texture, map);
    out[i] = PairingResult{partner.x, partner.y, back.x, back.y};
}

// Each traced pixel's partner in a frame's pairing over the checkerboard's
// traced lattice (metal/reuse/pairing.metal.h, lattice_partner), and its
// partner's; an untraced pixel's result is the pixel itself, twice.
// Bindings: the results at 0, the query (frame, texture, image size) at 1,
// the reuse textures at 2.
kernel void pt_lattice_pairing(device PairingResult* out [[buffer(0)]], constant PairingQuery& q [[buffer(1)]],
                               constant ushort* textures [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= q.width * q.height) {
        return;
    }
    const int2 pixel = int2(int(i % q.width), int(i / q.width));
    if (!serenity::contracts::checkerboard_traced(uint(pixel.x), uint(pixel.y), q.frame)) {
        out[i] = PairingResult{pixel.x, pixel.y, pixel.x, pixel.y};
        return;
    }
    const pairing::Framing map = pairing::framing(q.frame, q.texture);
    const int phase = int(q.frame & 1u);
    const int2 partner = pairing::lattice_partner(textures, pixel, q.texture, map, phase);
    const int2 back = pairing::lattice_partner(textures, partner, q.texture, map, phase);
    out[i] = PairingResult{partner.x, partner.y, back.x, back.y};
}

// A path drawn at each query's source pixel, shifted to its destination.
// Bindings: the results at 0, the queries at 1, the camera at 2, the
// structure at 3, the scene's block at 4, its glows at 5 and transforms at
// 6, the settings at 7, the queries' count at 8, the light tiles at 9, and
// MNEE's work list at 10 and its header at 11, a list of capacity 0: the
// trees' MNEE requests are counted and dropped, MNEE's paths being out of
// the path space every tree and shift samples (restir_pt.metal.h).
kernel void pt_shift(device ShiftResult* out [[buffer(0)]], const device ShiftQuery* queries [[buffer(1)]],
                     constant serenity::contracts::CameraData& camera [[buffer(2)]],
                     serenity::shaders::SceneStructure structure [[buffer(3)]],
                     constant serenity::gpu::SceneBlock& block [[buffer(4)]],
                     constant float* sphere_glows [[buffer(5)]],
                     constant serenity::contracts::Transform* transforms [[buffer(6)]],
                     constant serenity::contracts::PathReuseSettings& settings [[buffer(7)]],
                     constant uint& count [[buffer(8)]],
                     const device serenity::contracts::LightTileEntry* tiles [[buffer(9)]],
                     device serenity::contracts::MneeRequest* mnee_requests [[buffer(10)]],
                     device atomic_uint* mnee_header [[buffer(11)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const ShiftQuery q = queries[i];
    const uint2 size = uint2(q.width, q.height);
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    // A pixel's tree as the initial pass draws it: its reservoir and its
    // G-buffer record.
    const auto tree_at = [&](uint2 pixel) {
        PathNumbers numbers;
        const CameraRay ray = restir_pt::primary_ray(camera, pixel, q.frame, size, numbers);
        restir_pt::TreeGather tree =
            restir_pt::start_tree(settings, numbers, scene.shapes(), restir_pt::block_tile(tiles, pixel, q.frame),
                                  mnee::Queue{mnee_requests, mnee_header}, 0u);
        (void)path::walk_path(scene, path::walk_from(ray), numbers, tree);
        return tree;
    };
    const uint2 source = uint2(q.source_x, q.source_y);
    const uint2 destination = uint2(q.destination_x, q.destination_y);
    const serenity::contracts::PathReservoir reservoir = restir_pt::finish_tree(tree_at(source));
    const serenity::contracts::PrimaryHit record = tree_at(destination).primary;
    const CameraRay source_ray = restir_pt::primary_ray(camera, source, q.frame, size);
    const CameraRay destination_ray = restir_pt::primary_ray(camera, destination, q.frame, size);
    const shift::Shifted shifted = shift::shift_path(scene, shift::unpack_primary_hit(record), reservoir, settings);
    const shift::Shifted retraced =
        shift::shift_path(scene, shift::primary_hit(scene, destination_ray, settings.footprint), reservoir, settings);
    const shift::Primary source_hit = shift::primary_hit(scene, source_ray, settings.footprint);
    const shift::Primary destination_hit = shift::primary_hit(scene, destination_ray, settings.footprint);
    const float3 reconnection = serenity::contracts::path_rc(reservoir.flags) != 0u
                              ? decode_shape_point(scene.shapes(), reservoir.primitive,
                                                   serenity::contracts::ShapePoint{
                                                       reservoir.point, serenity::contracts::path_face(reservoir.flags)})
                              : float3(0.0f);
    out[i] = ShiftResult{reservoir.integrand,
                         reservoir.flags,
                         to_packed(shifted.integrand),
                         shifted.jacobian,
                         source_hit.here.surface.position,
                         shifted.shifted ? 1u : 0u,
                         destination_hit.here.surface.position,
                         reservoir.weight,
                         to_packed(reconnection),
                         reservoir.jacobian,
                         to_packed(retraced.integrand),
                         retraced.jacobian,
                         retraced.shifted ? 1u : 0u,
                         {0u, 0u, 0u}};
}

// The light tiles pt_shift's trees draw from, frame 0's. Bindings: as
// pt_shift binds the scene, the structure at 3, the block at 4, the glows
// at 5, the transforms at 6; the tiles it writes at 9.
kernel void pt_light_tiles(serenity::shaders::SceneStructure structure [[buffer(3)]],
                           constant serenity::gpu::SceneBlock& block [[buffer(4)]],
                           constant float* sphere_glows [[buffer(5)]],
                           constant serenity::contracts::Transform* transforms [[buffer(6)]],
                           device serenity::contracts::LightTileEntry* tiles [[buffer(9)]],
                           uint n [[thread_position_in_grid]]) {
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    if (n >= serenity::contracts::light_tiles * serenity::contracts::light_tile_size || scene.selection().count == 0u) {
        return;
    }
    tiles[n] = restir_pt::tile_entry(scene, n, 0u);
}

// The temporal merge on given inputs, its cap computed as the pass does
// (restir_pt_temporal.metal, restir_pt_temporal). Bindings: the results at
// 0, the queries at 1, their count at 2.
kernel void pt_temporal_merge(device PathMergeResult* out [[buffer(0)]], const device PathMergeQuery* queries [[buffer(1)]],
                              constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const PathMergeQuery q = queries[i];
    const float cap = path_merge::duplication_cap(q.cap, q.floor, q.exponent, q.duplication);
    const restir_pt::Merged merged =
        restir_pt::temporal_merge(q.own, q.last, float4(to_float3(q.here), q.here_jacobian),
                                  float4(to_float3(q.there), q.there_jacobian), cap, float2(q.u0, q.u1));
    out[i] = PathMergeResult{merged.reservoir, to_packed(merged.estimate), cap};
}

// The duplication map of an image of reservoirs, as the pass computes it
// (restir_pt_duplication.metal), dispatched as it is: whole groups.
// Bindings: the map at 0, the reservoirs at 1, the image's size at 2.
kernel void pt_duplication(device float* map [[buffer(0)]],
                           const device serenity::contracts::PathReservoir* finals [[buffer(1)]],
                           constant uint2& size [[buffer(2)]], uint2 pixel [[thread_position_in_grid]],
                           uint2 local [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup uint2 keys[duplication::span * duplication::span];
    duplication::load_keys(keys, finals, size, group * duplication::group, local.y * duplication::group + local.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pixel.x < size.x && pixel.y < size.y) {
        map[pixel.y * size.x + pixel.x] = duplication::score(keys, local);
    }
}
