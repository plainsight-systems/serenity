#pragma once

// Axis: Material (shader half of materials/coated.h).
//
// The coated BSDF's four questions, by core/materials/coated.h's steps: a
// smooth clear coat's mirror over a Lambert base. As every reflecting kind
// here, two-sided: `n` is the shading normal turned to wo's side
// (bsdf.cuh, facing), so wo is never below it.
//
// F(cos theta_o) is found once per question: sample() hands what it found
// to the base's f rather than evaluating it again (P.9).
//
// Translated line for line from Metal's metal/materials/coated.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/bsdf.h"
#include "cuda/device/layout.cuh"
#include "cuda/materials/fresnel.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"

namespace serenity {
namespace shaders {

// F(cos), the coat's reflectance from air into its ior.
__device__ inline float coat_reflectance(serenity::contracts::Bsdf bsdf,
                                         float cos_theta) {
  return fresnel(saturate(cos_theta), 1.0f / bsdf.ior).reflectance;
}

// Step 1's f for wi at cos_i above the surface, `through_o` = 1 - F(cos
// theta_o) given: light refracted in, scattered by the base and refracted
// out, its bounces under the coat summed.
__device__ inline float3 coated_base(serenity::contracts::Bsdf bsdf,
                                     float through_o, float cos_i) {
  const float3 rho = to_float3(bsdf.color);
  const float through = (1.0f - coat_reflectance(bsdf, cos_i)) * through_o;
  // 1 - rho F_in, as (1 - rho) + rho (1 - F_in): no rounded value taken
  // from 1, so finite while the escape is above 0 (materials/coated.h).
  return through * rho * inverse_pi /
         (bsdf.ior * bsdf.ior * ((1.0f - rho) + rho * bsdf.escape));
}

// Step 1: the base's f; 0 below the surface.
__device__ inline float3 coated_evaluate(serenity::contracts::Bsdf bsdf,
                                         float3 n, float3 wo, float3 wi) {
  const float cos_i = dot(wi, n);
  const float cos_o = dot(wo, n);
  if (cos_i <= 0.0f || cos_o <= 0.0f) {
    return make_float3(0.0f);
  }
  return coated_base(bsdf, 1.0f - coat_reflectance(bsdf, cos_o), cos_i);
}

// Step 3: the base's density as sample() draws it; the coat's is delta.
__device__ inline float coated_pdf(serenity::contracts::Bsdf bsdf, float3 n,
                                   float3 wo, float3 wi) {
  const float cos_i = dot(wi, n);
  if (cos_i <= 0.0f) {
    return 0.0f;
  }
  return (1.0f - coat_reflectance(bsdf, dot(wo, n))) * cos_i * inverse_pi;
}

// Step 2: the coat with probability F(cos theta_o), u.x below it: the
// first of its two lobes over u.x (contract 2). Else the base,
// cosine-weighted from u.yz. wo in the surface's plane has no sample (I.5):
// the coat's value F / cos would divide by its cosine of 0, and anything
// arriving there is weighed by that 0.
__device__ inline serenity::contracts::BsdfSample coated_sample(
    serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 u) {
  const float cos_o = dot(wo, n);
  if (cos_o <= 0.0f) {
    return serenity::contracts::BsdfSample{};
  }
  const float f_o = coat_reflectance(bsdf, cos_o);
  if (u.x < f_o) {
    // The coat's share spent choosing it: weight value |cos| / pdf = 1.
    return serenity::contracts::BsdfSample{
        to_packed(reflect(-wo, n)), f_o, to_packed(make_float3(f_o / cos_o)),
        serenity::contracts::lobe_reflection | serenity::contracts::lobe_delta};
  }
  const float3 wi = cosine_direction(n, yz(u));
  const float cos_i = dot(wi, n);
  if (cos_i <= 0.0f) {
    return serenity::contracts::BsdfSample{};  // on the horizon: no density
                                               // there
  }
  return serenity::contracts::BsdfSample{
      to_packed(wi), (1.0f - f_o) * cos_i * inverse_pi,
      to_packed(coated_base(bsdf, 1.0f - f_o, cos_i)),
      serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse};
}

}  // namespace shaders
}  // namespace serenity
