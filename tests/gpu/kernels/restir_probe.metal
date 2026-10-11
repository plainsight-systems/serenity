// Runs ReSTIR DI's shader pieces on their own for tests/gpu/restir_test.cpp:
// the equal-area octahedral map (metal/math/warp.metal.h), the packed BSDF
// (metal/materials/bsdf.metal.h), the light point (metal/lights/
// emitter.metal.h), the target (metal/integrator/target.metal.h), the
// reservoir (metal/reuse/reservoir.metal.h) and the shade's rule for a
// reservoir's test (metal/passes/restir_di/restir_di.metal.h). Bindings: the
// output at 0, the inputs after it in the order each kernel lists them
// (probes.h). A light kernel's lights are one sphere light, record 0, read
// through the emitter as the passes read them: the records, the spheres,
// the transforms, the glows and which light each shape is, at 1 to 5.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "core/passes/restir_di.h"
#include "metal/integrator/target.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/math/hash.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/passes/restir_di/restir_di.metal.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::contracts::Bsdf;
using serenity::contracts::LightPoint;
using serenity::tests::LightPointQuery;
using serenity::tests::LightPointResult;
using serenity::tests::SoftmaxDraw;
using serenity::tests::SoftmaxQuery;
using serenity::tests::TargetQuery;
using serenity::tests::TargetResult;
using serenity::tests::UnitDraws;
using serenity::tests::UnitTrip;

// Each direction's code, and the code decoded. Inputs: the directions, their
// count.
kernel void unit_trip(device UnitTrip* out [[buffer(0)]],
                      device const serenity::contracts::Float3* directions [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const uint code = encode_unit(to_float3(directions[i]));
    out[i] = UnitTrip{to_packed(decode_unit(code)), code};
}

// Draw i of a distribution over the sphere, from numbers hashed from i, as
// its code. Input: the draws' query.
kernel void unit_codes(device uint* out [[buffer(0)]],
                       constant UnitDraws& draws [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= draws.count) {
        return;
    }
    PathNumbers numbers = path_numbers(uint2(i & 0xffffu, i >> 16u), 0x5eed5u);
    const float2 u = next_numbers2(numbers);
    const float3 d = draws.distribution == serenity::tests::draw_cosine
                         ? cosine_direction(float3(0.0f, 0.0f, 1.0f), u)
                         : float3(metal::sqrt(metal::max(0.0f, 1.0f - (1.0f - 2.0f * u.x) * (1.0f - 2.0f * u.x))) *
                                      metal::cos(2.0f * M_PI_F * u.y),
                                  metal::sqrt(metal::max(0.0f, 1.0f - (1.0f - 2.0f * u.x) * (1.0f - 2.0f * u.x))) *
                                      metal::sin(2.0f * M_PI_F * u.y),
                                  1.0f - 2.0f * u.x);
    out[i] = encode_unit(d);
}

// Each Bsdf packed and unpacked. Inputs: the Bsdfs, their count.
kernel void bsdf_round_trip(device Bsdf* out [[buffer(0)]],
                            device const Bsdf* bsdfs [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = unpack_bsdf(pack_bsdf(bsdfs[i]));
    }
}

// The probes' one light, read as the passes read theirs.
inline Lights probe_lights(constant serenity::lights::LightRecord* records,
                           constant serenity::lights::SphereLightData* spheres,
                           constant serenity::contracts::Transform* transforms, constant float* glows,
                           constant uint* shape_lights) {
    return Lights{records, shape_lights, spheres, transforms, glows};
}

// From each query's point with its u: sample_light(), and draw_light_point()
// then evaluate_light_point() at the same point. Inputs: the lights (1 to
// 5), the queries, their count.
kernel void light_points(device LightPointResult* out [[buffer(0)]],
                         constant serenity::lights::LightRecord* records [[buffer(1)]],
                         constant serenity::lights::SphereLightData* spheres [[buffer(2)]],
                         constant serenity::contracts::Transform* transforms [[buffer(3)]],
                         constant float* glows [[buffer(4)]],
                         constant uint* shape_lights [[buffer(5)]],
                         device const LightPointQuery* queries [[buffer(6)]],
                         constant uint& count [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const Lights lights = probe_lights(records, spheres, transforms, glows, shape_lights);
    const float3 point = to_float3(queries[i].point);
    const float2 u = float2(queries[i].u0, queries[i].u1);
    const serenity::contracts::LightSample drawn = sample_light(lights, records[0], point, u);
    const LightPoint y = draw_light_point(lights, 0u, point, u);
    LightPointResult result{};
    result.sample_direction = drawn.direction;
    result.sample_distance = drawn.distance;
    result.sample_radiance = drawn.radiance;
    result.sample_pdf = drawn.pdf;
    result.light = y.light;
    if (serenity::contracts::holds_light(y)) {
        const serenity::contracts::LightReceived received = evaluate_light_point(lights, y, point);
        result.direction = received.direction;
        result.distance = received.distance;
        result.radiance = received.radiance;
        result.cos_light = received.cos_light;
        result.area_scale = received.area_scale;
        result.pdf = light_pdf(lights, records[0], point, to_float3(received.direction));
        result.area_density = light_area_density(result.pdf, received);
    }
    out[i] = result;
}

// At each query's surface, of `bsdf`, the target and its kin for light
// record 0's point in the query, the cone's density being light_pdf()'s
// toward it. Inputs: the lights (1 to 5), the queries, their count, the
// Bsdf.
kernel void target_at(device TargetResult* out [[buffer(0)]],
                      constant serenity::lights::LightRecord* records [[buffer(1)]],
                      constant serenity::lights::SphereLightData* spheres [[buffer(2)]],
                      constant serenity::contracts::Transform* transforms [[buffer(3)]],
                      constant float* glows [[buffer(4)]],
                      constant uint* shape_lights [[buffer(5)]],
                      device const TargetQuery* queries [[buffer(6)]],
                      constant uint& count [[buffer(7)]],
                      constant Bsdf& bsdf [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const Lights lights = probe_lights(records, spheres, transforms, glows, shape_lights);
    const TargetQuery query = queries[i];
    const RecordedSurface surface{to_float3(query.position), to_float3(query.wo), bsdf};
    const LightPoint y{serenity::contracts::light_valid | 0u, query.light_point};
    const serenity::contracts::LightReceived received = evaluate_light_point(lights, y, surface.position);
    const float density = light_pdf(lights, records[0], surface.position, to_float3(received.direction));
    out[i] = TargetResult{target(surface, received), candidate_weight(surface, received, query.selection * density),
                          light_area_density(density, received), received.cos_light,
                          to_packed(direct_light(surface, received, 1.0f)), received.distance};
}

// Stream i of query.streams: the scores streamed as candidates, weight and
// target exp(scale x score) (or 0), score j's identity light_valid | j,
// each kept by a number of the stream's own, then finished as resampled
// importance sampling ({1, M}). Input: the query.
kernel void softmax_streams(device SoftmaxDraw* out [[buffer(0)]],
                            constant SoftmaxQuery& query [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= query.streams) {
        return;
    }
    PathNumbers numbers = PathNumbers{uint2(i, query.seed), 0u};
    Streaming reservoir = empty_reservoir();
    for (uint n = 0; n < query.count; ++n) {
        const uint j = query.order == serenity::tests::order_reversed  ? query.count - 1u - n
                       : query.order == serenity::tests::order_rotated ? (n + i) % query.count
                                                                       : n;
        const float weight = query.zero != 0u ? 0.0f : metal::exp(query.scale * query.scores[j]);
        (void)stream_candidate(reservoir, Candidate{LightPoint{serenity::contracts::light_valid | j, 0u}, weight, weight},
                               next_number(numbers));
    }
    const serenity::contracts::Reservoir finished = finish(reservoir, Normalization{1.0f, float(reservoir.count)});
    out[i] = SoftmaxDraw{finished.sample.light, finished.count, finished.target, finished.weight,
                         reservoir.weight_sum, {0u, 0u, 0u}};
}

// The shade's rule (restir::takes_test) for each test word under
// `settings`: 1 where the reservoir's own test is taken for this pixel's
// now. Inputs: the words, their count, the settings.
kernel void takes_test_of(device uint* out [[buffer(0)]],
                          device const uint* tested [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant serenity::passes::RestirDi& settings [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = restir::takes_test(tested[i], settings) ? 1u : 0u;
    }
}
