#pragma once

// Axis: Integrator (ReSTIR PT's next event estimation).
//
// Next event estimation (NEE) at one vertex of ReSTIR PT's path tree by
// resampled importance sampling over the lights (ReSTIR PT Enhanced, Lin,
// Kettunen and Wyman 2026, §6.1, §7; its supplemental §5): M candidates,
// each a light chosen uniformly (P = 1 / N) and a point drawn on its cone
// from where the vertex's rays leave (path.metal.h, "One origin"), density
// p1 = P p_l in solid angle; each weighed by its unshadowed target
// p̂ = luminance(f |cos| L_e) over p1; one kept by the streaming pick;
// visibility tested for the kept one alone. M is nee_candidates at the
// primary hit and nee_candidates / B^2 at vertex B, at least 1 (Enhanced
// §7's inverse-square schedule). One candidate's draw and weight are here,
// the loop that
// streams them is the Pass's (metal/passes/restir_pt/restir_pt.metal.h),
// whose pick is Sample reuse's (metal/reuse/reservoir.metal.h, stream_pick):
// the Integrator includes no Sample reuse header (tools/check_boundaries.sh
// rule 5).
//
// What NEE contributes, in the primary sample space (supplemental §5): the
// path ending on the kept light has the integrand
//
//   F = w_nee f |cos| L_e T V / p1,   w_nee = M p1 / (M p1 + p_b)
//
// p_b the BSDF's density of the same direction, the M-sample balance
// heuristic over NEE and BSDF sampling (the 2022 code's evalMIS(), Balance,
// its default), as if the M candidates were all evaluated; and the
// unbiased contribution weight W_RIS p1 in place of 1, W_RIS = (1 / p̂(y))
// (1 / M) sum w_i. A ray the BSDF draws that strikes a light is weighed by
// the other half, w_bsdf = p_b / (M p1 + p_b), with the M of the vertex it
// left; so for every direction both strategies draw, the weights sum to 1.
// T is the medium's transmittance over the shadow ray (contract 12).
//
// ReSTIR PT draws its candidates' lights from the frame's light tiles
// (contracts/light_tile.h; Enhanced §6.1): draw_tile_candidate(), the same
// draw from a 32-byte packed entry of the tile its pixel's block holds, in
// place of four dependent reads of the light's record, sphere, transform
// and glow (metal/lights/emitter.metal.h, place_light). Measured 4.7% faster
// for the initial pass on the marbles
// (docs/research/2026-10-11-restir-pt-phase2.md), so kept.
//
// Cost per vertex: M candidates of 4 numbers, a tile entry's 32 bytes, a
// point on its cone, a BSDF evaluation each; one shadow ray. 32 at the
// primary hit, so the initial pass's cost is dominated by its first vertex
// (measured: docs/research/2026-10-11-restir-pt-phase1.md, phase2.md).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/math/srgb.metal.h"

namespace serenity {
namespace shaders {
namespace nee {

// The candidates at the path's vertex `vertex`, 1 the primary hit.
inline uint candidate_count(uint candidates, uint index) {
  return metal::max(1u, candidates / (index * index));
}

// One candidate, drawn and weighed (above): drawn by the light's own
// sampling, a direction in its cone (contract 3), not as a stored point
// (contract 4), so a candidate costs no encoding of a point it will most
// often not keep; the kept one's point is found once, by kept_point().
struct Candidate {
  uint light;  // the light record's index
  // its direction, distance and radiance from the origin
  serenity::contracts::LightSample sample;
  float density;  // p1 = P p_l, solid angle
  float target;   // p̂(y), unshadowed
};

inline Candidate no_candidate() {
  return Candidate{0u, serenity::contracts::LightSample{}, 0.0f, 0.0f};
}

// The unshadowed f |cos| L_e at `here` of light arriving along `wi` with
// `radiance`, 0 where wi leaves on wo's other side (path.metal.h, "One
// origin").
inline float3 unshadowed(Shading here, float3 wi, float3 radiance) {
  if (!leave_alike(here.surface, wi, here.wo)) {
    return float3(0.0f);
  }
  return bsdf_evaluate(here.bsdf, here.wo, wi) *
         metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) * radiance;
}

// The same for a light's answer at a stored point (contract 4), 0 where the
// light faces away.
inline float3 unshadowed(Shading here,
                         serenity::contracts::LightReceived light) {
  return light.cos_light > 0.0f ? unshadowed(here, to_float3(light.direction),
                                             to_float3(light.radiance))
                                : float3(0.0f);
}

// A candidate drawn from a light tile's entry (contracts/light_tile.h):
// the draw contract 3's sample_light() makes from the light the entry was
// packed from, its density the selection's `probability` of the entry's
// light (its power over the frame's total, light_selection/
// power_light.metal.h) times the cone's.
inline Candidate draw_tile_candidate(serenity::contracts::LightTileEntry entry,
                                     float probability, Shading here,
                                     float3 origin, float2 u) {
  const serenity::contracts::LightSample sample =
      sample_tile_light(entry, origin, u);
  Candidate candidate{entry.ids & serenity::contracts::tile_index_mask, sample,
                      probability * sample.pdf, 0.0f};
  if (candidate.density > 0.0f) {
    candidate.target = luminance(unshadowed(here, to_float3(sample.direction),
                                            to_float3(sample.radiance)));
  }
  return candidate;
}

// The kept candidate's point on its light (contract 4), seen from `origin`
// again from the stored point, as a shift that reconnects to it sees it: the
// point, and the light's answer there (its cosine and distance for the
// point's area density).
struct KeptPoint {
  serenity::contracts::LightPoint point;
  serenity::contracts::LightReceived light;
};

inline KeptPoint kept_point(Lights lights, Candidate kept, float3 origin) {
  const PlacedLight light = place_light(lights, kept.light);
  const float3 reached =
      origin + kept.sample.distance * to_float3(kept.sample.direction);
  const serenity::contracts::LightPoint point =
      light_point_at(light, kept.light, reached);
  return KeptPoint{point, evaluate_light_point(light, point, origin)};
}

// The M-sample balance heuristic's weights (above), each for the strategy
// named, from M, p1 and p_b of one direction. Defined where both are 0: a
// direction neither draws is weighed 0.
inline float light_weight(uint m, float light_density, float bsdf_density) {
  const float light = float(m) * light_density;
  const float sum = light + bsdf_density;
  return sum > 0.0f ? light / sum : 0.0f;
}
inline float bsdf_weight(uint m, float light_density, float bsdf_density) {
  const float sum = float(m) * light_density + bsdf_density;
  return sum > 0.0f ? bsdf_density / sum : 0.0f;
}

// Whether nothing blocks the light along `direction` for `distance` from
// `origin` (path.metal.h, step 5's shadow ray): among the surfaces alone
// (contract 19, the occluders), so no firefly blocks it, the light's own
// shape `primitive` among them. The one shadow ray of ReSTIR PT's initial
// paths and of its shifts' reconnections to a light, so a shift tests the
// visibility its path was sampled with.
inline bool visible(SceneStructure structure, Shapes shapes, float3 origin,
                    float3 direction, float distance, uint primitive) {
  return !occluded(structure, shapes,
                   metal::raytracing::ray(origin, direction, 0.0f, distance));
}

inline bool visible(SceneStructure structure, Shapes shapes, float3 origin,
                    serenity::contracts::LightReceived light) {
  return visible(structure, shapes, origin, to_float3(light.direction),
                 light.distance, light.primitive);
}

}  // namespace nee
}  // namespace shaders
}  // namespace serenity
