// Runs the packed forms ReSTIR PT stores on their own for
// tests/gpu/encoding_test.cpp: the equal-area octahedral map
// (metal/math/warp.metal.h), the packed BSDF (metal/materials/
// bsdf.metal.h) and the light point (metal/lights/emitter.metal.h).
// Bindings: the output at 0, the inputs after it in the order each kernel
// lists them (encoding_probe.h). The light kernel's light is one sphere
// light, record 0, read through the emitter as the passes read them: the
// records, the spheres, the transforms, the glows and which light each
// shape is, at 1 to 5 (gpu/support/probe_light.h).

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "encoding_probe.h"
#include "metal/lights/emitter.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/sampler/sampler.metal.h"

using namespace serenity::shaders;
using serenity::contracts::Bsdf;
using serenity::contracts::LightPoint;
using serenity::tests::LightPointQuery;
using serenity::tests::LightPointResult;
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

// From each query's point with its u: sample_light(); then the point that
// direction reaches on the light kept as its identity, light_point_at(), and
// evaluated from the same point, evaluate_light_point() and
// light_point_pdf(), as ReSTIR PT keeps a NEE candidate's point
// (metal/integrator/nee.metal.h, kept_point). Inputs: the light (1 to 5),
// the queries, their count.
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
    const Lights lights{records, shape_lights, spheres, transforms, glows};
    const float3 point = to_float3(queries[i].point);
    const float2 u = float2(queries[i].u0, queries[i].u1);
    const serenity::contracts::LightSample drawn = sample_light(lights, records[0], point, u);
    LightPointResult result{};
    result.sample_direction = drawn.direction;
    result.sample_distance = drawn.distance;
    result.sample_radiance = drawn.radiance;
    result.sample_pdf = drawn.pdf;
    if (drawn.pdf > 0.0f) {
        const PlacedLight light = place_light(lights, 0u);
        const float3 reached = point + drawn.distance * to_float3(drawn.direction);
        const LightPoint y = light_point_at(light, 0u, reached);
        const serenity::contracts::LightReceived received = evaluate_light_point(light, y, point);
        result.light = y.light;
        result.direction = received.direction;
        result.distance = received.distance;
        result.radiance = received.radiance;
        result.cos_light = received.cos_light;
        result.area_scale = received.area_scale;
        result.pdf = light_point_pdf(light, point, received);
    }
    out[i] = result;
}
