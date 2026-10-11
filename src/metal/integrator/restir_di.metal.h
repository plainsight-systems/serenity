#pragma once

// Axis: Integrator (ReSTIR DI's light transport).
//
// What light ReSTIR DI's image holds, apart from how its samples are
// resampled and shared: the surface its direct light is estimated at, one
// candidate drawn toward a light and weighed there, whether a light point
// is visible from that surface, and the shade's transport, the direct light
// from a chosen sample and the path continued for the indirect. Each is the
// naive path tracer's own transport where it can be (path.metal.h,
// camera_segment() and continue_path()), and the target where it weighs
// (target.metal.h), so ReSTIR and the reference share every line of light
// transport but the direct light at the first rough surface
// (logical-overview.md, principle 3; ES.3).
//
// Where it sits (change-axes.md: families meet at contracts; decided
// 2026-10-11 by the review of f86e7ae, which found these functions in the
// Pass's file, deciding the image's light transport there):
//
//   Integrator    this file, path.metal.h and target.metal.h. Sees a
//                 reservoir only as contract 14's data
//                 (core/contracts/reservoir.h); includes no metal/reuse/
//                 header, and no Pass.
//   Sample reuse  metal/reuse/: streaming, finishing and merging reservoirs,
//                 and the rule for taking a stored visibility test. Includes
//                 contracts and the Sampler; no Integrator, no Pass.
//   Pass          metal/passes/restir_di/restir_di.metal.h and the passes'
//                 kernels: composition and dispatch. Loops M times, asks this
//                 file to draw and weigh a candidate, Sample reuse to stream
//                 it, this file for visibility and the shade's transport;
//                 binds and barriers. Includes both.
//
// tools/check_boundaries.sh holds the first two: metal/integrator/ includes
// no metal/reuse/ or metal/passes/ header, and metal/reuse/ no
// metal/integrator/ or metal/passes/ one, nor a core/passes/ one.
//
// The functions, in the order ReSTIR's steps ask them
// (restir_di.metal.h numbers the steps):
//
//   first_surface()   step 1. The camera's path, as the path tracer runs
//                     it (camera_segment()), to its first surface with a lobe
//                     that is not delta: its surface record (contract 5: the
//                     point, wo and Bsdf packed, as a G-buffer holds them),
//                     its path state (throughput, the light found on the way,
//                     media, the surface's count, the numbers' counter) and,
//                     with last frame's transforms, where its point was last
//                     frame. No such surface: the record's kind none, and
//                     the path state's light the pixel's.
//   draw_candidate()  step 2, one light candidate: a light chosen as the
//                     naive estimator chooses one, uniformly
//                     (light_selection/uniform_light.metal.h), placed once
//                     (lights/emitter.metal.h, place_light: its record, data,
//                     transform and glow, which drawing, evaluating and the
//                     density then read), a point on it drawn as
//                     sample_light() draws a direction and evaluated at its
//                     decoded point (contracts/light_sample.h), and its
//                     target and its weight p̂ / p (target.metal.h), both in
//                     the light's own area, p the two strategies' density
//                     (target.metal.h, candidate_density). What Sample
//                     reuse streams.
//   draw_bsdf_candidate()  step 2, one BSDF candidate (RTXDI's BRDF sample,
//                     Phase B): a direction drawn by the surface's BSDF,
//                     one ray to the nearest surface, and where it strikes a
//                     light, that point stored as a light point
//                     (light_point_at()) and weighed as a light candidate is
//                     (weigh()), so both strategies weigh one sample by one
//                     density.
//   visible()         steps 3 and 6: one shadow ray from a recorded surface
//                     to a light point's answer. The one visibility test
//                     every step makes, so a sample is visible from a
//                     surface by every step or by none.
//   unshadowed()      step 6: the direct light at the surface from a sample
//                     and its weight W, f |cos| L_e G T W, without V
//                     (target.metal.h, direct_light; T the path's medium's
//                     transmittance), and the light's answer for the ray
//                     that tests V.
//   continue_from()   step 6: the pixel's radiance, the path state's light
//                     plus its throughput times the direct light, then the
//                     path continued from the record's surface as the
//                     naive path tracer continues it (continue_path(),
//                     step 5 there left out). The reservoir's estimate is
//                     the direct light at q, weighed against no BSDF sample,
//                     so the bounce from q counts no firefly it strikes off a
//                     lobe that is not delta (path.metal.h, "ReSTIR's first
//                     surface"); one its coat's delta lobe reaches is
//                     counted in full; from the next surface on, the
//                     continuation weighs light and BSDF sampling by MIS as
//                     the path tracer does, being its loop (principle 7).
//
// Every function asks its questions of the surface unpacked from its record
// (target.metal.h, unpack_record), so the surface the candidates are
// weighed at is the one the shade shades. The initial pass unpacks it once
// for its candidates and their ray. The shade unpacks it for its direct
// light, and continue_from() again where the continuation starts, rather
// than the shade holding the 72-byte unpacked surface across its
// visibility query: the query is in the shade's code whether or not a
// pixel traces it, so what is alive across it costs registers in every
// graph, and two unpacks (two unit-vector decodes) cost less (decided
// 2026-10-11 by measurement, against the review of f86e7ae's unpack-once:
// docs/research/2026-10-10-restir-di-layout.md, "The review of f86e7ae").
//
// Cost: first_surface() is the camera segment's (path.metal.h); a candidate
// is one placed light (76 bytes of logical loads), one point drawn and
// evaluated, one density, one target and weight (a BSDF evaluation and a
// dozen flops), and with BSDF candidates one BSDF density more; a BSDF
// candidate one BSDF sample, one ray to the nearest surface and, where it
// strikes a light, one placed light, its point encoded and evaluated, and
// two BSDF densities and one evaluation; visible() one ray; unshadowed() one placed light and one
// BSDF evaluation; continue_from() the path tracer's continuation.
// restir_di.metal.h counts them a pixel.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/light_sample.h"
#include "core/contracts/surface_interaction.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/transform.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/integrator/target.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/scene/scene_block.metal.h"

namespace serenity {
namespace shaders {
namespace restir {

// The scene, its lights chosen among uniformly, as the path tracer's.
using Scene = SceneView<UniformLight>;

// Contract 12's medium index in the path state's 16 bits, and back
// (contracts/surface_record.h): no_medium is 0xffff there, and every other
// index is under it, which the renderer checks of the scene (I.5).
inline uint pack_medium(uint medium) {
    return medium == serenity::contracts::no_medium ? serenity::contracts::path_no_medium : medium;
}

inline uint unpack_medium(uint packed) {
    const uint medium = packed & serenity::contracts::path_counter_mask;
    return medium == serenity::contracts::path_no_medium ? serenity::contracts::no_medium : medium;
}

// Step 1's result for one pixel.
struct Surfaces {
    serenity::contracts::SurfaceRecord record;
    serenity::contracts::PathState state;
    float3 previous_position;  // where the record's point was last frame; unset without last_transforms
};

// Where `surface`'s point was last frame: its object-space point placed by
// its shape's transform among `last_transforms`, last frame's (contract
// 10: s R p + T).
inline float3 previous_position(constant serenity::contracts::Transform* last_transforms,
                                serenity::contracts::SurfaceInteraction surface) {
    const serenity::contracts::Transform placed = last_transforms[surface.primitive];
    return transform_translation(placed) + transform_direction(placed, to_float3(surface.object_position));
}

// Step 1: the camera's segment along `ray`, as a record and a path state;
// `numbers` the path's stream, which the state's counter records.
// `last_transforms`, last frame's shape transforms in a graph with a
// temporal merge and null otherwise, places the surface's point as last
// frame placed its shape (previous_position()). A path that ends before
// such a surface leaves the record zero, its bsdf's kind none, and its
// light in the state.
inline Surfaces first_surface(Scene scene, CameraRay ray, constant serenity::contracts::Transform* last_transforms,
                              thread PathNumbers& numbers) {
    const path::Segment segment = path::camera_segment(scene, ray, numbers);
    Surfaces found{};
    found.state.radiance = to_packed(segment.L);
    found.state.counts = numbers.dimension & serenity::contracts::path_counter_mask;
    found.state.media = serenity::contracts::path_no_medium |
                        (serenity::contracts::path_no_medium << serenity::contracts::path_interior_shift);
    if (!segment.at_surface) {
        return found;
    }
    const Shading here = segment.here;
    found.record = surface_record(here);
    found.state.throughput = to_packed(segment.beta);
    found.state.eta_scale = segment.eta_scale;
    found.state.counts |= (segment.surface << serenity::contracts::path_surface_shift) |
                          (here.surface.flags << serenity::contracts::path_flags_shift);
    found.state.media =
        pack_medium(here.medium) | (pack_medium(here.surface.interior) << serenity::contracts::path_interior_shift);
    if (last_transforms != nullptr) {
        found.previous_position = previous_position(last_transforms, here.surface);
    }
    return found;
}

// One candidate drawn and weighed (step 2): its light point, what the light
// there gives the surface (kept beside the stream for the survivor's
// shadow ray, so nothing is evaluated again), its target p̂ and its weight
// p̂ / p. A candidate whose point holds no light (drawn from inside its
// light) weighs 0, its target 0: a draw whose sample does not exist.
struct Weighed {
    serenity::contracts::LightPoint sample;     // y
    serenity::contracts::LightReceived light;   // y's answer at the surface; set where y holds a light
    float target;                               // p̂_q(y)
    float weight;                               // p̂_q(y) / p(y), finite and at least 0
};

// No candidate: a draw whose sample does not exist, weighing 0.
inline Weighed no_candidate() {
    return Weighed{serenity::contracts::LightPoint{0u, 0u}, serenity::contracts::LightReceived{}, 0.0f, 0.0f};
}

// `sample`, a point on `light` chosen with probability `probability`, as
// either strategy's candidate at `here` weighs it: its answer, its target
// and its weight p̂ / p, p the two strategies' density (target.metal.h,
// candidate_density), so a light point is weighed the same whichever
// strategy drew it. The light strategy's density is asked of the point
// (light_point_pdf()), not of the direction toward it, whose test against
// the light's rim rounding fails for some points on the rim: a BSDF ray
// strikes the rim as often as anywhere, and a point there weighed as if no
// light candidate could draw it is counted twice, which the BSDF
// candidates' floor test in tests/gpu/restir_test.cpp finds. Precondition: holds_light(sample).
inline Weighed weigh(PlacedLight light, serenity::contracts::LightPoint sample, RecordedSurface here,
                     float probability, CandidateCounts counts) {
    Weighed candidate{sample, evaluate_light_point(light, sample, here.position), 0.0f, 0.0f};
    const float3 wi = to_float3(candidate.light.direction);
    const StrategyDensities densities{probability * light_point_pdf(light, here.position, candidate.light),
                                      counts.bsdf > 0u ? bsdf_pdf(here.bsdf, here.wo, wi) : 0.0f};
    candidate.target = target(here, candidate.light);
    candidate.weight =
        candidate_weight(here, candidate.light, candidate_density(counts, densities, candidate.light.distance));
    return candidate;
}

// Step 2, one light candidate at `here`: its light chosen from `lights` by
// `selection` with the number `choose`, its point drawn with `point_numbers`
// (the Pass's to draw, in the order the stream gives them), weighed against
// both strategies by `counts`. Precondition: the scene has a light.
inline Weighed draw_candidate(Lights lights, UniformLight selection, RecordedSurface here, CandidateCounts counts,
                              float choose, float2 point_numbers) {
    const SelectedIndex chosen = select_light_index(selection, choose);
    const PlacedLight light = place_light(lights, chosen.index);
    const serenity::contracts::LightPoint sample = draw_light_point(light, chosen.index, here.position, point_numbers);
    if (!serenity::contracts::holds_light(sample)) {
        return no_candidate();
    }
    return weigh(light, sample, here, chosen.probability, counts);
}

// A BSDF candidate's second half: the light record `index` its ray struck
// at `reached`, as a light point weighed by weigh(); none where the light
// point is beyond the BSDF's reach from `here` (target.metal.h,
// bsdf_reach), which RTXDI tests again at the stored point. No ray: what
// the ray found is its argument, so a probe can weigh any point.
inline Weighed weigh_struck(Lights lights, UniformLight selection, RecordedSurface here, CandidateCounts counts,
                            uint index, float3 reached) {
    const PlacedLight light = place_light(lights, index);
    const Weighed candidate =
        weigh(light, light_point_at(light, index, reached), here, light_probability(selection, light.record), counts);
    const float bsdf_density = bsdf_pdf(here.bsdf, here.wo, to_float3(candidate.light.direction));
    return candidate.light.distance <= bsdf_reach(bsdf_density) ? candidate : no_candidate();
}

// Step 2, one BSDF candidate at `here` (target.metal.h, RTXDI's BRDF
// sample): a direction drawn by its BSDF with `numbers`, traced from where
// a ray leaves the surface (integrator/surface.metal.h, leave) as far as
// the BSDF reaches, and the light it strikes first weighed (weigh_struck());
// none from a delta lobe, which the target does not weigh, nor where the
// ray strikes nothing or a shape that is not a light. One ray to the
// nearest surface, always traced where the lobe is not delta.
inline Weighed draw_bsdf_candidate(Scene scene, RecordedSurface here, CandidateCounts counts, float3 numbers) {
    const serenity::contracts::BsdfSample drawn = bsdf_sample(here.bsdf, here.wo, numbers);
    if (drawn.pdf <= 0.0f || (drawn.lobe & serenity::contracts::lobe_delta) != 0u) {
        return no_candidate();
    }
    serenity::contracts::SurfaceInteraction leaving{};
    leaving.position = to_packed(here.position);
    leaving.geometric_normal = here.bsdf.normal;
    const metal::raytracing::ray along = leave(leaving, to_float3(drawn.direction), bsdf_reach(drawn.pdf));
    const Hit hit = trace(scene.structure, scene.shapes(), along);
    if (!hit.found) {
        return no_candidate();
    }
    const uint index = light_index_at(scene.lights(), hit.primitive);
    if (index == serenity::lights::no_light) {
        return no_candidate();
    }
    return weigh_struck(scene.lights(), scene.selection(), here, counts, index,
                        along.origin + hit.t * along.direction);
}

// Whether `light`, as evaluate_light_point() gave it to `from`'s point, is
// visible from `from`: one shadow ray against `scene`'s structure, leaving
// along the surface's unpacked shading normal (contracts/surface_record.h),
// ignoring the light's own shape. The ray leaves as every ray leaves a
// surface (integrator/surface.metal.h, leave: ES.3), from a surface holding
// only the point and the normal leave() reads.
inline bool visible(Scene scene, RecordedSurface from, serenity::contracts::LightReceived light) {
    serenity::contracts::SurfaceInteraction leaving{};
    leaving.position = to_packed(from.position);
    leaving.geometric_normal = from.bsdf.normal;
    return !occluded(scene.structure, scene.shapes(), leave(leaving, to_float3(light.direction), light.distance),
                     light.primitive);
}

// Step 6's direct light before visibility: f |cos θ_x| L_e G x T x W at
// `here` from `sample` with contribution weight `weight` (> 0), T the
// transmittance of `medium`, the path's there, over the distance to the
// light; and the light's answer, for the shadow ray that tests V.
// Precondition: holds_light(sample).
struct Unshadowed {
    float3 radiance;
    serenity::contracts::LightReceived light;
};

inline Unshadowed unshadowed(Scene scene, RecordedSurface here, uint medium, serenity::contracts::LightPoint sample,
                             float weight) {
    const serenity::contracts::LightReceived light = evaluate_light_point(scene.lights(), sample, here.position);
    return Unshadowed{direct_light(here, light, weight) * transmittance(scene.media(), medium, light.distance),
                      light};
}

// Step 6's radiance: the path state's light, plus its throughput times
// `direct`, the direct light at the record's surface (0 where there is
// none, or it is hidden), then the continuation from that surface,
// `record` unpacked here (above) and rebuilt as continue_path() reads it
// (path.metal.h): the point and the normal as the geometric normal, the
// flags and the interior from the path state; and the Shading's BSDF, wo
// and medium. `numbers` the path's stream, from the state's counter.
inline float3 continue_from(Scene scene, serenity::contracts::SurfaceRecord record,
                            serenity::contracts::PathState state, float3 direct, thread PathNumbers& numbers) {
    const RecordedSurface here = unpack_record(record);
    const uint medium = unpack_medium(state.media);
    const float3 beta = to_float3(state.throughput);
    serenity::contracts::SurfaceInteraction surface{};
    surface.position = record.position;
    surface.geometric_normal = here.bsdf.normal;
    surface.flags = state.counts >> serenity::contracts::path_flags_shift;
    surface.interior = unpack_medium(state.media >> serenity::contracts::path_interior_shift);
    const path::Segment segment{true,
                                Shading{surface, here.bsdf, here.wo, medium},
                                to_float3(state.radiance) + beta * direct,
                                beta,
                                state.eta_scale,
                                (state.counts >> serenity::contracts::path_surface_shift) &
                                    serenity::contracts::path_surface_mask};
    return path::continue_path(scene, segment, numbers);
}

}  // namespace restir
}  // namespace shaders
}  // namespace serenity
