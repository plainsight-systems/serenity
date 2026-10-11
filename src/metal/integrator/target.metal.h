#pragma once

// Axis: Integrator (the target function).
//
// What "contribution" means for ReSTIR DI: the target p̂_q(y) a light sample
// y is weighed by at surface q, ignoring what may be in the way. The
// Integrator owns it (change-axes.md: the target function belongs with the
// light transport it stands for); the candidates, every merge and every Z
// test call this one definition (ES.3), so none can weigh a sample one way
// and normalize it another.
//
// Depends on contracts 2, 4 and 5 and the Integrator's own Shading; nothing
// of Sample reuse. The merges reach it only through the Frame the
// composition passes them (metal/reuse/unbiased.metal.h;
// metal/passes/restir_di/restir_di.metal.h), so a change to the merges
// changes nothing here, and a change here none of their headers.
//
// The target, for surface record q (contracts/surface_record.h) and the
// light at y as evaluate_light_point() gives it (contracts/light_sample.h):
//
//   p̂_q(y) = luminance(f(wo, wi) |cos θ_x| L_e) x max(cos θ_y, 0) a / d^2
//
// wi and d the direction and distance from q's point to y's, θ_x between wi
// and q's shading normal, θ_y at the light, between its normal at y and
// -wi, and a the light's area scale. That is f |cos| L_e G with G the
// geometry term into the measure the sample lives in, area on the light in
// its own coordinates (light_sample.h), dω = cos θ_y a dA / d^2: one sample
// means one thing at every surface and in every frame, so a weight from
// last frame's surface, or a neighbor's, is in the same units as this one's
// (Bitterli et al. 2020 shift samples between pixels unchanged: the shift
// is the identity, in a measure all pixels share). Luminance by Rec. 709's
// weights, 0.2126, 0.7152 and 0.0722, the radiance's primaries being sRGB's.
//
// Any target with the integrand's support makes the estimator unbiased; this
// one is the integrand without visibility and transmittance, which makes it
// the best that needs no ray (the paper's choice). Its support is the
// integrand's: f > 0, L_e > 0 and y on the side of the light that faces q;
// what visibility removes is accounted for by Z (metal/reuse/
// unbiased.metal.h), and a medium's transmittance, never 0, by the shade.
//
// The density a candidate is drawn with, in the same measure: Light
// selection's probability of its light, times the emitter's density per
// solid angle of the direction toward y (contract 3, light_pdf), times
// cos θ_y a / d^2. A candidate's weight p̂ / p then needs neither cos θ_y
// nor d: they cancel, and the weight is luminance(f |cos θ_x| L_e) over the
// selection probability and the solid-angle density, finite at the light's
// rim, where cos θ_y goes to 0 in both. The implementation computes the
// cancelled form; the header says what it is equal to.
//
// The shade's estimate of the direct light at q, given the final reservoir's
// sample and weight W (metal/passes/restir_di/restir_di.metal.h, step 6):
//
//   L_direct = f(wo, wi) |cos θ_x| L_e max(cos θ_y, 0) a / d^2 x T x W
//
// V absent: by the merges' invariant, a reservoir's sample is visible from
// its surface or W is 0 (unbiased.metal.h). T is the transmittance of the
// medium the path is in over d (contract 12), 1 in air.
//
// The light sample is evaluated at the point its identity holds,
// everywhere, the candidates' weights included: the point as drawn, kept
// exactly (contracts/light_sample.h), so the density a candidate is weighed
// by is its own point's, the measure the reference integrates.
//
// Pure functions of their arguments, inline, nothing read from memory: the
// record and the light's answer are in registers (F.8: pure functions, so
// any pass may evaluate them in any order and get the same).
//
// Cost, per evaluation: one BSDF evaluation (contract 2: a few dozen flops
// for the conductor's GGX, fewer for the others), the light's answer
// (evaluate_light_point(): a transform of the point, a normalization, and
// 76 bytes of the light's data loaded), and a dozen flops more.
// metal/passes/restir_di/restir_di.metal.h counts the evaluations a pixel
// makes.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/surface_record.h"
#include "metal/integrator/surface.metal.h"

namespace serenity {
namespace shaders {

// The record made from a surface being shaded: its resolved Bsdf, point and
// wo (contracts/surface_record.h). The initial pass makes it; every later
// pass reads the Bsdf from it as it is.
inline serenity::contracts::SurfaceRecord surface_record(Shading here);

// Rec. 709 luminance of a linear sRGB radiance.
inline float luminance(float3 radiance);

// p̂_q(y): `record` at q, `light` what the light at y gives q's point.
inline float target(serenity::contracts::SurfaceRecord record, serenity::contracts::LightReceived light);

// A density per unit solid angle at q's point turned into one per unit of
// the light's own area at y: times cos θ_y a / d^2; 0 where cos θ_y <= 0.
inline float light_area_density(float solid_angle_density, serenity::contracts::LightReceived light);

// The direct light at q from y, without visibility, times `weight`: the
// shade's estimate with W; the caller multiplies by its medium's T.
inline float3 direct_light(serenity::contracts::SurfaceRecord record, serenity::contracts::LightReceived light,
                           float weight);

}  // namespace shaders
}  // namespace serenity
