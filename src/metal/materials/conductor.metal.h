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
// glossy.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/materials/conductor.h"
#include "metal/device/layout.metal.h"
#include "metal/sampler/sampler.metal.h"

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
    const float l = n_v * metal::sqrt(a2 + (1.0f - a2) * n_l * n_l);
    const float v = n_l * metal::sqrt(a2 + (1.0f - a2) * n_v * n_v);
    return 2.0f * n_l * n_v / (l + v);
}

inline float3 schlick(float3 f0, float cos_theta) {
    const float m = metal::pow(1.0f - metal::saturate(cos_theta), 5.0f);
    return f0 + (1.0f - f0) * m;
}

// The BRDF times the cosine to `l`: what multiplies the radiance arriving
// along `l` per unit solid angle.
inline float3 conductor_reflectance(serenity::materials::ConductorData conductor, float alpha, float3 n, float3 v,
                                    float3 l) {
    const float n_l = metal::dot(n, l);
    const float n_v = metal::dot(n, v);
    if (n_l <= 0.0f || n_v <= 0.0f) {
        return float3(0.0f);
    }
    const float3 h = metal::normalize(v + l);
    const float d = ggx_d(metal::saturate(metal::dot(n, h)), alpha);
    const float g = smith_g2(n_l, n_v, alpha);
    const float3 f = schlick(to_float3(conductor.f0), metal::dot(v, h));
    return f * (d * g / (4.0f * n_v));
}

// A microfacet normal visible from `v`, in world space, from a point `u` in
// the unit square.
inline float3 sample_visible_normal(float3 n, float3 v, float alpha, float2 u) {
    float3 t;
    float3 b;
    basis(n, t, b);
    const float3 ve = float3(metal::dot(v, t), metal::dot(v, b), metal::dot(v, n));
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
    return ne.x * t + ne.y * b + ne.z * n;
}

// `n` is the normal turned to wo's side.
inline float3 conductor_evaluate(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float n_o = metal::dot(n, wo);
    const float n_i = metal::dot(n, wi);
    if (n_o <= 0.0f || n_i <= 0.0f) {
        return float3(0.0f);
    }
    const float3 h = metal::normalize(wo + wi);
    const float d = ggx_d(metal::saturate(metal::dot(n, h)), bsdf.alpha);
    const float g = smith_g2(n_i, n_o, bsdf.alpha);
    return schlick(to_float3(bsdf.color), metal::dot(wo, h)) * (d * g / (4.0f * n_o * n_i));
}

inline float conductor_pdf(serenity::contracts::Bsdf bsdf, float3 n, float3 wo, float3 wi) {
    const float n_o = metal::dot(n, wo);
    if (n_o <= 0.0f || metal::dot(n, wi) <= 0.0f) {
        return 0.0f;
    }
    const float3 h = metal::normalize(wo + wi);
    return smith_g1(n_o, bsdf.alpha) * ggx_d(metal::saturate(metal::dot(n, h)), bsdf.alpha) / (4.0f * n_o);
}

inline serenity::contracts::BsdfSample conductor_sample(serenity::contracts::Bsdf bsdf, float3 n, float3 wo,
                                                        float2 u) {
    serenity::contracts::BsdfSample s;
    s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_glossy;
    s.pdf = 0.0f;
    s.direction = to_packed(float3(0.0f));
    s.value = to_packed(float3(0.0f));
    if (metal::dot(n, wo) <= 0.0f) {
        return s;
    }
    const float3 h = sample_visible_normal(n, wo, bsdf.alpha, u);
    const float3 wi = metal::reflect(-wo, h);
    if (metal::dot(n, wi) <= 0.0f) {
        return s;  // reflected below the horizon: no sample
    }
    s.direction = to_packed(wi);
    s.pdf = conductor_pdf(bsdf, n, wo, wi);
    s.value = to_packed(conductor_evaluate(bsdf, n, wo, wi));
    return s;
}

}  // namespace shaders
}  // namespace serenity
