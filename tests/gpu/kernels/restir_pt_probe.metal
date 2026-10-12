// ReSTIR PT's pieces on their own (tests/gpu/restir_pt_test.cpp): the
// reconnection criteria on given pairs, the paired neighbors of a frame,
// and a path tree's path shifted to another pixel, each through the
// functions the passes call.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/camera.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/transform.h"
#include "metal/integrator/reconnection.metal.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"
#include "metal/reuse/pairing.metal.h"
#include "restir_pt_probe.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
using serenity::tests::CriteriaQuery;
using serenity::tests::CriteriaResult;
using serenity::tests::PairingQuery;
using serenity::tests::PairingResult;
using serenity::tests::ShiftQuery;
using serenity::tests::ShiftResult;

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

kernel void pt_shift(device ShiftResult* out [[buffer(0)]], const device ShiftQuery* queries [[buffer(1)]],
                     constant serenity::contracts::CameraData& camera [[buffer(2)]],
                     primitive_acceleration_structure structure [[buffer(3)]],
                     constant serenity::gpu::SceneBlock& block [[buffer(4)]],
                     constant float* sphere_glows [[buffer(5)]],
                     constant serenity::contracts::Transform* transforms [[buffer(6)]],
                     constant serenity::contracts::PathReuseSettings& settings [[buffer(7)]],
                     constant uint& count [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const ShiftQuery q = queries[i];
    const uint2 size = uint2(q.width, q.height);
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    PathNumbers numbers;
    const CameraRay source_ray =
        restir_pt::primary_ray(camera, uint2(q.source_x, q.source_y), q.frame, size, numbers);
    restir_pt::TreeGather tree = restir_pt::start_tree(settings, numbers, scene.shapes());
    (void)path::walk_path<path::Start::camera, path::Stop::at_end>(scene, path::walk_from(source_ray), Shading{}, 0u,
                                                                   numbers, tree);
    const serenity::contracts::PathReservoir reservoir = restir_pt::finish_tree(tree);
    const CameraRay destination_ray =
        restir_pt::primary_ray(camera, uint2(q.destination_x, q.destination_y), q.frame, size);
    const shift::Shifted shifted = shift::shift_path(scene, destination_ray, reservoir, settings);
    const shift::Primary source_hit = shift::primary_hit(scene, source_ray);
    const shift::Primary destination_hit = shift::primary_hit(scene, destination_ray);
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
                         reservoir.jacobian};
}
