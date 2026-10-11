// Runs the path tracer's multiple importance sampling (metal/integrator/
// path.metal.h: next_event_weight(), emission_weight(), steps 5 and 3) on
// its own for tests/gpu/path_test.cpp, with the densities each step asks:
// the selection's (light_selection/uniform_light.metal.h), the emitter's
// (contract 3) and the BSDF's (contract 2). Bindings: the output at 0, then
// the lights at 1 to 5 (the records, the spheres, the transforms, the
// glows and which light each shape is: one sphere light, record 0, read
// through the emitter as the passes read it), then the inputs each kernel
// lists (path_probe.h).

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/transform.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "metal/integrator/path.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/math/hash.metal.h"
#include "path_probe.h"

using namespace serenity::shaders;
using serenity::contracts::Bsdf;
using serenity::tests::DirectIntegral;
using serenity::tests::DirectQuery;
using serenity::tests::MisDraw;
using serenity::tests::MisQuery;

// Draw i of 2 x query.draws: the first query.draws toward the light, as
// step 5 draws (sample_light()), the rest by the BSDF, as step 6 draws
// (bsdf_sample()), each from numbers hashed from i; and for its direction,
// the weight each step gives it, from the density each step has for it:
//
//   - drawn toward the light: w_l from the sample's own density, times the
//     selection's 1 / N (select_light()), against the BSDF's pdf() of it,
//     as step 5; w_b from the BSDF's pdf() of it, the density step 6 would
//     have drawn it with, against the light's light_pdf() of it, as step 3.
//   - drawn by the BSDF: w_b from the sample's own pdf, against
//     light_pdf(), as step 3; w_l from light_pdf() of it, the density step
//     5 would have drawn it with, against the BSDF's pdf(), as step 5; 0
//     where light_pdf() is 0, since step 5 draws no such direction.
//
// Inputs: the lights (1 to 5), the query, the Bsdf.
kernel void mis_weights(device MisDraw* out [[buffer(0)]],
                        constant serenity::lights::LightRecord* records [[buffer(1)]],
                        constant serenity::lights::SphereLightData* spheres [[buffer(2)]],
                        constant serenity::contracts::Transform* transforms [[buffer(3)]],
                        constant float* glows [[buffer(4)]],
                        constant uint* shape_lights [[buffer(5)]],
                        constant MisQuery& query [[buffer(6)]],
                        constant Bsdf& bsdf [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= 2u * query.draws) {
        return;
    }
    const Lights lights{records, shape_lights, spheres, transforms, glows};
    const UniformLight step_5{records, query.light_count};
    const UniformLight step_3{records, query.struck_light_count};
    const serenity::lights::LightRecord light = records[0];
    const float3 point = to_float3(query.point);
    const float3 wo = to_float3(query.wo);
    const uint h = pcg_hash(i ^ 0x2545f491u);
    const float3 u = float3(unit_float(h), unit_float(pcg_hash(h)), unit_float(pcg_hash(h ^ 0x9e3779b9u)));
    MisDraw draw{};
    if (i < query.draws) {
        // As step 5 chooses its light: the probability select_light() gives.
        const SelectedLight chosen = select_light(step_5, 0.0f);
        const serenity::contracts::LightSample sample = sample_light(lights, light, point, u.xy);
        const float3 wi = to_float3(sample.direction);
        draw.light_density = chosen.probability * sample.pdf;
        draw.bsdf_density = bsdf_pdf(bsdf, wo, wi);
        draw.light_weight = draw.light_density > 0.0f ? path::next_event_weight(draw.light_density, bsdf, wo, wi) : 0.0f;
        draw.bsdf_weight = draw.bsdf_density > 0.0f
                               ? path::emission_weight(step_3, lights, light, point, wi, draw.bsdf_density)
                               : 0.0f;
        draw.by_bsdf = 0u;
    } else {
        const serenity::contracts::BsdfSample sample = bsdf_sample(bsdf, wo, u);
        const float3 wi = to_float3(sample.direction);
        draw.bsdf_density = sample.pdf;
        draw.light_density = light_probability(step_5, light) * light_pdf(lights, light, point, wi);
        draw.bsdf_weight =
            sample.pdf > 0.0f ? path::emission_weight(step_3, lights, light, point, wi, sample.pdf) : 0.0f;
        draw.light_weight = draw.light_density > 0.0f ? path::next_event_weight(draw.light_density, bsdf, wo, wi) : 0.0f;
        draw.by_bsdf = 1u;
    }
    out[i] = draw;
}

// At each query's point, of `bsdf` seen from its wo, the light's integral
// f |cos| L_e over its cone, red channel, by the midpoint rule over a grid
// of query.grid x query.grid cells of u, which sample_light() maps to the
// cone uniformly in solid angle, so the integral is the cells' mean times
// the cone's solid angle, 1 / the sample's pdf; and the share of it that
// step 5's light samples carry once weighted, the mean of the same terms
// times w_l over their mean. Unshadowed: the probe's scene has nothing
// between the point and the light. Inputs: the lights (1 to 5), the
// queries, their count, the Bsdf.
kernel void light_integral(device DirectIntegral* out [[buffer(0)]],
                           constant serenity::lights::LightRecord* records [[buffer(1)]],
                           constant serenity::lights::SphereLightData* spheres [[buffer(2)]],
                           constant serenity::contracts::Transform* transforms [[buffer(3)]],
                           constant float* glows [[buffer(4)]],
                           constant uint* shape_lights [[buffer(5)]],
                           device const DirectQuery* queries [[buffer(6)]],
                           constant uint& count [[buffer(7)]],
                           constant Bsdf& bsdf [[buffer(8)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const Lights lights{records, shape_lights, spheres, transforms, glows};
    const DirectQuery query = queries[i];
    const UniformLight selection{records, query.light_count};
    const serenity::lights::LightRecord light = records[0];
    const float3 point = to_float3(query.point);
    const float3 wo = to_float3(query.wo);
    const uint n = query.grid;
    float sum = 0.0f;
    float weighted = 0.0f;
    float solid_angle = 0.0f;
    for (uint cell = 0; cell < n * n; ++cell) {
        const float2 u = (float2(cell % n, cell / n) + 0.5f) / float(n);
        const serenity::contracts::LightSample sample = sample_light(lights, light, point, u);
        const float3 wi = to_float3(sample.direction);
        const float term = bsdf_evaluate(bsdf, wo, wi).x * metal::abs(metal::dot(wi, to_float3(bsdf.normal))) *
                           to_float3(sample.radiance).x;
        sum += term;
        const float light_density = light_probability(selection, light) * sample.pdf;
        weighted += term > 0.0f ? term * path::next_event_weight(light_density, bsdf, wo, wi) : 0.0f;
        solid_angle = 1.0f / sample.pdf;
    }
    out[i] = DirectIntegral{sum / float(n * n) * solid_angle, sum > 0.0f ? weighted / sum : 0.0f, {0u, 0u}};
}
