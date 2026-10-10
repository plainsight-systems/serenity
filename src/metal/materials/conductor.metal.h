#pragma once

// Axis: Material (shader half of materials/conductor.h).
//
// The GGX microfacet reflection of a metal: the distribution D, the
// height-correlated Smith masking G2 (and G1 for one direction), Schlick's
// Fresnel from f0, and sampling of the visible normals (Heitz, "Sampling the
// GGX Distribution of Visible Normals", JCGT 2018), for which the weight of
// a reflected sample is F G2 / G1(v). alpha = roughness^2. Directions are
// unit vectors on the surface's side; n is the surface normal.
//
// Its BSDF (contracts/bsdf.h): f = F D G2 / (4 |n.wo| |n.wi|), sampled by the
// visible normals, pdf = G1(wo) D(h) / (4 |n.wo|). One lobe: reflection,
// glossy. Both are written once, given the half vector h: evaluate() and
// pdf() find it as normalize(wo + wi), sample() has it, the visible normal
// it drew, and does not find it, or D, again (P.9).

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "metal/device/layout.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

inline float ggx_d(float n_h, float alpha) {
    const float a2 = alpha * alpha;
    const float k = n_h * n_h * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * k * k);
}

inline float smith_g1(float n_v, float alpha) {
    const float a2 = alpha * alpha;
    return 2.0f * n_v / (n_v + metal::sqrt(a2 + (1.0f - a2) * n_v * n_v));
}

inline float smith_g2(float n_l, float n_v, float alpha) {
    const float a2 = alpha * alpha;
    const float masked_l = n_v * metal::sqrt(a2 + (1.0f - a2) * n_l * n_l);
    const float masked_v = n_l * metal::sqrt(a2 + (1.0f - a2) * n_v * n_v);
    return 2.0f * n_l * n_v / (masked_l + masked_v);
}

// (1 - cos)^5 as products: metal::pow is exp2 and log2 under fast math.
inline float3 schlick(float3 f0, float cos_theta) {
    const float x = 1.0f - metal::saturate(cos_theta);
    const float x2 = x * x;
    return f0 + (1.0f - f0) * (x2 * x2 * x);
}

// A microfacet normal visible from `v`, in world space, from a point `u` in
// the unit square.
inline float3 sample_visible_normal(float3 n, float3 v, float alpha, float2 u) {
    const Tangents f = tangents(n);
    const float3 ve = float3(metal::dot(v, f.t), metal::dot(v, f.b), metal::dot(v, n));
    const float3 vh = metal::normalize(float3(alpha * ve.x, alpha * ve.y, ve.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 0.0f ? float3(-vh.y, vh.x, 0.0f) * metal::rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = metal::cross(vh, t1);
    const float r = metal::sqrt(u.x);
    const float phi = 2.0f * M_PI_F * u.y;
    const float p1 = r * metal::cos(phi);
    const float s = 0.5f * (1.0f + vh.z);
    const float p2 = (1.0f - s) * metal::sqrt(1.0f - p1 * p1) + s * r * metal::sin(phi);
    const float3 nh = p1 * t1 + p2 * t2 + metal::sqrt(metal::max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    const float3 ne = metal::normalize(float3(alpha * nh.x, alpha * nh.y, metal::max(0.0f, nh.z)));
    return ne.x * f.t + ne.y * f.b + ne.z * n;
}

// f, given h, with n_o = n.wo and n_i = n.wi both above 0.
inline float3 conductor_f(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 h, float n_o, float n_i) {
    const float d = ggx_d(metal::saturate(metal::dot(n, h)), bsdf.alpha);
    const float g = smith_g2(n_i, n_o, bsdf.alpha);
    return schlick(to_float3(bsdf.color), metal::dot(wo, h)) * (d * g / (4.0f * n_o * n_i));
}

// The pdf, given h, with n_o = n.wo above 0.
inline float conductor_density(serenity::contracts::Bsdf bsdf, float3 n, float3 h, float n_o) {
    return smith_g1(n_o, bsdf.alpha) * ggx_d(metal::saturate(metal::dot(n, h)), bsdf.alpha) / (4.0f * n_o);
}

// `n` is the normal turned to wo's side.
inline float3 conductor_evaluate(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float n_o = metal::dot(n, wo);
    const float n_i = metal::dot(n, wi);
    if (n_o <= 0.0f || n_i <= 0.0f) {
        return float3(0.0f);
    }
    return conductor_f(bsdf, n, wo, metal::normalize(wo + wi), n_o, n_i);
}

inline float conductor_pdf(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float n_o = metal::dot(n, wo);
    if (n_o <= 0.0f || metal::dot(n, wi) <= 0.0f) {
        return 0.0f;
    }
    return conductor_density(bsdf, n, metal::normalize(wo + wi), n_o);
}

inline serenity::contracts::BsdfSample conductor_sample(serenity::contracts::Bsdf bsdf, float3 n, float3 wo,
                                                        float2 u) {
    const float n_o = metal::dot(n, wo);
    if (n_o <= 0.0f) {
        return serenity::contracts::BsdfSample{};
    }
    const float3 h = sample_visible_normal(n, wo, bsdf.alpha, u);
    const float3 wi = metal::reflect(-wo, h);
    const float n_i = metal::dot(n, wi);
    if (n_i <= 0.0f) {
        return serenity::contracts::BsdfSample{};  // reflected below the horizon: no sample
    }
    return serenity::contracts::BsdfSample{to_packed(wi), conductor_density(bsdf, n, h, n_o),
                                           to_packed(conductor_f(bsdf, n, wo, h, n_o, n_i)),
                                           serenity::contracts::lobe_reflection | serenity::contracts::lobe_glossy};
}

}  // namespace shaders
}  // namespace serenity
