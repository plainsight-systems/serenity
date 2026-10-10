#pragma once

// Axis: Material (shader half of materials/coated.h).
//
// The coated BSDF's four questions, by core/materials/coated.h's steps: a
// smooth clear coat's mirror over a Lambert base. As every reflecting kind
// here, two-sided: `n` is the shading normal turned to wo's side
// (bsdf.metal.h, facing), so wo is never below it.
//
// F(cos theta_o) is found once per question: sample() hands what it found
// to the base's f rather than evaluating it again (P.9).

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "metal/device/layout.metal.h"
#include "metal/materials/fresnel.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

// F(cos), the coat's reflectance from air into its ior.
inline float coat_reflectance(serenity::contracts::Bsdf bsdf, float cos_theta) {
    return fresnel(metal::saturate(cos_theta), 1.0f / bsdf.ior).reflectance;
}

// Step 1's f for wi at cos_i above the surface, `through_o` = 1 - F(cos
// theta_o) given: light refracted in, scattered by the base and refracted
// out, its bounces under the coat summed.
inline float3 coated_base(serenity::contracts::Bsdf bsdf, float through_o, float cos_i) {
    const float3 rho = to_float3(bsdf.color);
    const float through = (1.0f - coat_reflectance(bsdf, cos_i)) * through_o;
    // 1 - rho F_in, as (1 - rho) + rho (1 - F_in): no rounded value taken
    // from 1, so finite while the escape is above 0 (materials/coated.h).
    return through * rho * M_1_PI_F / (bsdf.ior * bsdf.ior * ((1.0f - rho) + rho * bsdf.escape));
}

// Step 1: the base's f; 0 below the surface.
inline float3 coated_evaluate(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float cos_i = metal::dot(wi, n);
    const float cos_o = metal::dot(wo, n);
    if (cos_i <= 0.0f || cos_o <= 0.0f) {
        return float3(0.0f);
    }
    return coated_base(bsdf, 1.0f - coat_reflectance(bsdf, cos_o), cos_i);
}

// Step 3: the base's density as sample() draws it; the coat's is delta.
inline float coated_pdf(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float cos_i = metal::dot(wi, n);
    if (cos_i <= 0.0f) {
        return 0.0f;
    }
    return (1.0f - coat_reflectance(bsdf, metal::dot(wo, n))) * cos_i * M_1_PI_F;
}

// Step 2: the coat with probability F(cos theta_o), u.x below it: the
// first of its two lobes over u.x (contract 2). Else the base,
// cosine-weighted from u.yz. wo in the surface's plane has no sample (I.5):
// the coat's value F / cos would divide by its cosine of 0, and anything
// arriving there is weighed by that 0.
inline serenity::contracts::BsdfSample coated_sample(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 u) {
    const float cos_o = metal::dot(wo, n);
    if (cos_o <= 0.0f) {
        return serenity::contracts::BsdfSample{};
    }
    const float f_o = coat_reflectance(bsdf, cos_o);
    if (u.x < f_o) {
        // The coat's share spent choosing it: weight value |cos| / pdf = 1.
        return serenity::contracts::BsdfSample{to_packed(metal::reflect(-wo, n)), f_o, to_packed(float3(f_o / cos_o)),
                                               serenity::contracts::lobe_reflection |
                                                   serenity::contracts::lobe_delta};
    }
    const float3 wi = cosine_direction(n, u.yz);
    const float cos_i = metal::dot(wi, n);
    if (cos_i <= 0.0f) {
        return serenity::contracts::BsdfSample{};  // on the horizon: no density there
    }
    return serenity::contracts::BsdfSample{to_packed(wi), (1.0f - f_o) * cos_i * M_1_PI_F,
                                           to_packed(coated_base(bsdf, 1.0f - f_o, cos_i)),
                                           serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse};
}

}  // namespace shaders
}  // namespace serenity
