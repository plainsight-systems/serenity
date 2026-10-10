#pragma once

// Axis: Material (shader half of materials/coated.h).
//
// The coated BSDF's four questions, by core/materials/coated.h's steps: a
// smooth clear coat's mirror over a Lambert base. As every reflecting kind
// here, two-sided: `n` is the shading normal turned to wo's side
// (bsdf.metal.h, facing), so wo is never below it.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "metal/device/layout.metal.h"
#include "metal/materials/dielectric.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

// F(cos), the coat's reflectance from air into its ior.
inline float coat_reflectance(serenity::contracts::Bsdf bsdf, float cos_theta) {
    float cos_t;
    return fresnel_reflectance(metal::saturate(cos_theta), 1.0f / bsdf.ior, cos_t);
}

// Step 1: the base's f, light refracted in, scattered by the base and
// refracted out, its bounces under the coat summed; 0 below the surface.
inline float3 coated_evaluate(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float cos_i = metal::dot(wi, n);
    const float cos_o = metal::dot(wo, n);
    if (cos_i <= 0.0f || cos_o <= 0.0f) {
        return float3(0.0f);
    }
    const float3 rho = to_float3(bsdf.color);
    const float through = (1.0f - coat_reflectance(bsdf, cos_i)) * (1.0f - coat_reflectance(bsdf, cos_o));
    return through * rho * M_1_PI_F / (bsdf.ior * bsdf.ior * (1.0f - rho * bsdf.internal));
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
// cosine-weighted from u.yz.
inline serenity::contracts::BsdfSample coated_sample(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 u) {
    const float cos_o = metal::dot(wo, n);
    const float f_o = coat_reflectance(bsdf, cos_o);
    serenity::contracts::BsdfSample s;
    if (u.x < f_o) {
        // The coat's share spent choosing it: weight value |cos| / pdf = 1.
        s.direction = to_packed(2.0f * cos_o * n - wo);
        s.pdf = f_o;
        s.value = to_packed(float3(f_o / cos_o));
        s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_delta;
        return s;
    }
    const float3 wi = cosine_direction(n, u.yz);
    s.direction = to_packed(wi);
    s.pdf = coated_pdf(bsdf, n, wo, wi);
    s.value = to_packed(coated_evaluate(bsdf, n, wo, wi));
    s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse;
    return s;
}

}  // namespace shaders
}  // namespace serenity
