// Runs the path tracer's multiple importance sampling (metal/integrator/
// path.metal.h: next_event_origin(), handed_on(), next_event_weight(),
// emission_weight() and power_heuristic(), steps 5, 6 and 3) on its own
// for tests/gpu/path_test.cpp, with the densities each step asks, from the
// points each asks them from (integrator/surface.metal.h, leave()):
// the selection's (light_selection/uniform_light.metal.h), the emitter's
// (contract 3) and the BSDF's (contract 2). Bindings: the output at 0, then
// the lights at 1 to 5 (the records, the spheres, the transforms, the
// glows and which light each shape is: one sphere light, record 0, read
// through the emitter as the passes read it), then the inputs each kernel
// lists (path_probe.h); heuristic_weights reads no light, its inputs from 1.

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
using serenity::tests::HeuristicQuery;
using serenity::tests::HeuristicWeights;
using serenity::tests::MisDraw;
using serenity::tests::MisQuery;

// Draw i of 2 x query.draws: the first query.draws toward the light, as
// step 5 draws (sample_light() from next_event_origin(), keeping only a
// direction that leaves alike with wo), the rest by the BSDF, as step 6
// draws (bsdf_sample(), its density handed on by handed_on()), each from
// numbers hashed from i; and for its direction, the weight each step gives
// it, from the density each step has for it, asked from the point each
// step asks it from:
//
//   - step 5's density of a direction: the selection's 1 / N times the
//     light's density from next_event_origin() (the sample's own pdf for
//     its own draw, light_pdf() for a bounce's), 0 where the direction does
//     not leave alike with wo, as step 5 does not draw there; w_l from it
//     against the BSDF's pdf(), as next_event_weight() weighs;
//   - step 6's density of a direction: for its own draw what handed_on()
//     gives (counted_in_full where step 5 does not draw the direction,
//     then w_b = 1), for a light sample's the BSDF's pdf() of it; w_b from
//     it by emission_weight(), the light's density asked from the point the
//     bounce along it leaves from, leave()'s, as steps 8 and 3 ask it.
//
// With light_from_point 1, step 5 draws from the surface's point instead,
// as 7751a65 did, and its density of a bounce's direction is asked there.
//
// Inputs: the lights (1 to 5), the query, the Bsdf.
kernel void mis_weights(
    device MisDraw* out [[buffer(0)]],
    constant serenity::lights::LightRecord* records [[buffer(1)]],
    constant serenity::lights::SphereLightData* spheres [[buffer(2)]],
    constant serenity::contracts::Transform* transforms [[buffer(3)]],
    constant float* glows [[buffer(4)]],
    constant uint* shape_lights [[buffer(5)]],
    constant MisQuery& query [[buffer(6)]], constant Bsdf& bsdf [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= 2u * query.draws) {
    return;
  }
  const Lights lights{records, shape_lights, spheres, transforms, glows};
  const UniformLight step_5{records, query.light_count};
  const UniformLight step_3{records, query.struck_light_count};
  const serenity::lights::LightRecord light = records[0];
  serenity::contracts::SurfaceInteraction surface{};
  surface.position = query.point;
  surface.geometric_normal = query.normal;
  const float3 wo = to_float3(query.wo);
  const Shading here{surface, bsdf, wo, serenity::contracts::no_medium};
  const float3 light_origin = query.light_from_point != 0u
                                  ? to_float3(surface.position)
                                  : path::next_event_origin(surface, wo);
  const uint h = pcg_hash(i ^ 0x2545f491u);
  const float3 u = float3(unit_float(h), unit_float(pcg_hash(h)),
                          unit_float(pcg_hash(h ^ 0x9e3779b9u)));
  MisDraw draw{};
  if (i < query.draws) {
    // As step 5 chooses its light: the probability select_light() gives.
    const SelectedLight chosen = select_light(step_5, 0.0f);
    const serenity::contracts::LightSample sample =
        sample_light(lights, light, light_origin, u.xy);
    const float3 wi = to_float3(sample.direction);
    draw.light_density =
        leave_alike(surface, wi, wo) ? chosen.probability * sample.pdf : 0.0f;
    draw.bsdf_density = bsdf_pdf(bsdf, wo, wi);
    draw.light_weight =
        draw.light_density > 0.0f
            ? path::next_event_weight(draw.light_density, bsdf, wo, wi)
            : 0.0f;
    draw.bsdf_weight = draw.bsdf_density > 0.0f
                           ? path::emission_weight(step_3, lights, light,
                                                   leave(surface, wi).origin,
                                                   wi, draw.bsdf_density)
                           : 0.0f;
    draw.by_bsdf = 0u;
  } else {
    const serenity::contracts::BsdfSample sample = bsdf_sample(bsdf, wo, u);
    const float3 wi = to_float3(sample.direction);
    draw.bsdf_density = sample.pdf;
    draw.light_density = leave_alike(surface, wi, wo)
                             ? light_probability(step_5, light) *
                                   light_pdf(lights, light, light_origin, wi)
                             : 0.0f;
    draw.bsdf_weight =
        sample.pdf > 0.0f ? path::emission_weight(step_3, lights, light,
                                                  leave(surface, wi).origin, wi,
                                                  path::handed_on(sample, here))
                          : 0.0f;
    draw.light_weight =
        draw.light_density > 0.0f
            ? path::next_event_weight(draw.light_density, bsdf, wo, wi)
            : 0.0f;
    draw.by_bsdf = 1u;
  }
  out[i] = draw;
}

// For each query, the path tracer's power_heuristic() both ways, and the
// directly squared form as a control (path_probe.h, HeuristicWeights).
// Inputs: the queries, their count.
kernel void heuristic_weights(device HeuristicWeights* out [[buffer(0)]],
                              device const HeuristicQuery* queries
                              [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= count) {
    return;
  }
  const HeuristicQuery q = queries[i];
  const float d2 = q.drawn * q.drawn;
  out[i] = HeuristicWeights{path::power_heuristic(q.drawn, q.other),
                            path::power_heuristic(q.other, q.drawn),
                            d2 / (d2 + q.other * q.other), 0u};
}
