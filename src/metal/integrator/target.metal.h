#pragma once

// Axis: Integrator (the target function).
//
// What "contribution" means for ReSTIR DI: the target p̂_q(y) a light sample
// y is weighed by at surface q, ignoring what may be in the way. The
// Integrator owns it (change-axes.md: the target function belongs with the
// light transport it stands for); the candidates, every merge and every
// normalization's domain target call this one definition (ES.3), so none
// can weigh a sample one way and normalize it another.
//
// Depends on contracts 2, 4 and 5, Material's packed form and the shared
// unit-vector map, and the Integrator's own Shading; nothing of Sample
// reuse. The merges reach it only through the Frame the
// composition passes them (metal/reuse/unbiased.metal.h;
// metal/passes/restir_di/restir_di.metal.h), so a change to the merges
// changes nothing here, and a change here none of their headers.
//
// The target, for surface record q (contracts/surface_record.h), unpacked,
// and the light at y as evaluate_light_point() gives it at y's decoded point
// (contracts/light_sample.h):
//
//   p̂_q(y) = luminance(f(wo, wi) |cos θ_x| L_e) x max(cos θ_y, 0) a / d^2
//
// f the record's unpacked Bsdf at its unpacked wo (contracts/bsdf.h,
// unpack_bsdf), wi and d the direction and distance from q's point to y's,
// θ_x between wi and q's shading normal, θ_y at the light, between its normal
// at y and -wi, and a the light's area scale. That is f |cos| L_e G with G
// the geometry term into the measure the sample lives in, area on the light
// in its own coordinates (light_sample.h), dω = cos θ_y a dA / d^2: one
// sample means one thing at every surface and in every frame, so a weight
// from last frame's surface, or a neighbor's, is in the same units as this
// one's (Bitterli et al. 2020 shift samples between pixels unchanged: the
// shift is the identity, in a measure all pixels share). Luminance by
// Rec. 709's weights, 0.2126, 0.7152 and 0.0722, the radiance's primaries
// being sRGB's.
//
// Any target with the integrand's support makes the estimator unbiased; this
// one is the integrand without visibility and transmittance, which makes it
// the best that needs no ray (the paper's choice). Its support is the
// integrand's: f > 0, L_e > 0 and y on the side of the light that faces q;
// what visibility removes is accounted for by the merges' normalization
// (metal/reuse/unbiased.metal.h), and a medium's transmittance, never 0, by
// the shade.
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
//   L_direct = f(wo, wi) |cos θ_x| L_e max(cos θ_y, 0) a / d^2 x V x T x W
//
// V the sample's visibility from q, which the shade has from its own
// shadow ray, from a recent test it reuses, or, in raytraced mode, from the
// merges' invariant (metal/reuse/unbiased.metal.h): the caller's, applied
// by not calling this where V is 0. T is the transmittance of the medium
// the path is in over d (contract 12), 1 in air. The shade evaluates the
// same unpacked surface the target weighed, so the target is the
// integrand without V and T exactly, at every pixel.
//
// The light sample is evaluated at the point its identity holds,
// everywhere, the candidates' weights included: the decoded point
// (contracts/light_sample.h), so the density a candidate is weighed by is
// its own point's. That point is its cell's representative, which is the
// quantization's small bias, stated there.
//
// Pure functions of their arguments, inline, nothing read from memory: the
// surface and the light's answer are in registers (F.8: pure functions, so
// any pass may evaluate them in any order and get the same).
//
// Cost, per evaluation: one BSDF evaluation (contract 2: a few dozen flops
// for the conductor's GGX, fewer for the others), the light's answer
// (evaluate_light_point(): the point's decoding, a transform, a
// normalization, and 76 bytes of the light's data loaded, once per placed
// light), and a dozen flops more. Unpacking a record (decoding wo and the
// normal, widening four halves: two square roots and a few dozen flops) is
// paid once per record a pass loads, by unpack_record(), not once per
// evaluation, where the caller holds the unpacked surface; the shade pays
// it twice, once for its direct light and once where its continuation
// starts, rather than hold the surface across its visibility query
// (metal/integrator/restir_di.metal.h); the merges' Frame, which is given
// records, unpacks per evaluation. Each is ALU traded for registers.
// metal/passes/restir_di/restir_di.metal.h counts the evaluations a pixel
// makes.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/surface_record.h"
#include "metal/integrator/surface.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

// The record made from a surface being shaded: its point, wo encoded and
// its resolved Bsdf packed (contracts/surface_record.h; contracts/bsdf.h,
// pack_bsdf). The initial pass makes it.
inline serenity::contracts::SurfaceRecord surface_record(Shading here) {
    return serenity::contracts::SurfaceRecord{here.surface.position, encode_unit(here.wo), pack_bsdf(here.bsdf)};
}

// A record unpacked: the surface every question about the pixel is asked
// of, from the initial pass's candidates to the shade (contracts/
// surface_record.h, "What the packing changes").
struct RecordedSurface {
    float3 position;
    float3 wo;                           // unit, decoded
    serenity::contracts::Bsdf bsdf;      // unpack_bsdf()
};
inline RecordedSurface unpack_record(serenity::contracts::SurfaceRecord record) {
    return RecordedSurface{to_float3(record.position), decode_unit(record.wo), unpack_bsdf(record.bsdf)};
}

// Rec. 709 luminance of a linear sRGB radiance.
inline float luminance(float3 radiance) {
    return metal::dot(radiance, float3(0.2126f, 0.7152f, 0.0722f));
}

// f(wo, wi) |cos θ_x| L_e at `surface` for `light`: the factor the target,
// a candidate's weight and the direct light share (ES.3).
inline float3 scattered(RecordedSurface surface, serenity::contracts::LightReceived light) {
    const float3 wi = to_float3(light.direction);
    return bsdf_evaluate(surface.bsdf, surface.wo, wi) * metal::abs(metal::dot(wi, to_float3(surface.bsdf.normal))) *
           to_float3(light.radiance);
}

// G's factor into the light's own area: cos θ_y a / d^2, 0 where cos θ_y
// <= 0 (the far side of the light gives nothing).
inline float area_factor(serenity::contracts::LightReceived light) {
    return light.cos_light > 0.0f ? light.cos_light * light.area_scale / (light.distance * light.distance) : 0.0f;
}

// p̂_q(y): `surface` q, `light` what the light at y gives q's point.
inline float target(RecordedSurface surface, serenity::contracts::LightReceived light) {
    const float g = area_factor(light);
    return g > 0.0f ? luminance(scattered(surface, light)) * g : 0.0f;
}

// A density per unit solid angle at q's point turned into one per unit of
// the light's own area at y: times cos θ_y a / d^2; 0 where cos θ_y <= 0.
inline float light_area_density(float solid_angle_density, serenity::contracts::LightReceived light) {
    return solid_angle_density * area_factor(light);
}

// A candidate's weight p̂ / p, p = `solid_angle_density` x cos θ_y a /
// d^2, the density the candidate was drawn with per unit solid angle at q,
// its light's selection probability times the emitter's (one number, so a
// caller cannot pass the two factors in each other's places: I.24), in
// the cancelled form above: luminance(f |cos θ_x| L_e) /
// solid_angle_density, 0 where cos θ_y <= 0 or the density is 0, finite at
// the light's rim, where target() and light_area_density() both go to 0.
// What the reservoir streams (metal/reuse/reservoir.metal.h, step 2).
inline float candidate_weight(RecordedSurface surface, serenity::contracts::LightReceived light,
                              float solid_angle_density) {
    return light.cos_light > 0.0f && solid_angle_density > 0.0f
               ? luminance(scattered(surface, light)) / solid_angle_density
               : 0.0f;
}

// The direct light at q from y, without visibility, times `weight`: the
// shade's estimate with W, where the caller has found y visible; the caller
// multiplies by its medium's T.
inline float3 direct_light(RecordedSurface surface, serenity::contracts::LightReceived light, float weight) {
    const float g = area_factor(light);
    return g > 0.0f ? scattered(surface, light) * (g * weight) : float3(0.0f);
}

}  // namespace shaders
}  // namespace serenity
