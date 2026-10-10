#pragma once

// Axis: Material (shader half of materials/dielectric.h).
//
// What a smooth dielectric boundary does to a ray arriving along `direction`
// at a surface with outward normal `normal`: the side it arrives from, the
// Fresnel reflectance for unpolarized light (the mean of the s and p terms),
// and the reflected and refracted directions. Under total internal
// reflection the reflectance is 1 and there is no refracted direction.
//
// With cos_i the cosine of the angle of incidence and eta = n_from / n_to
// (1 / ior entering, ior leaving):
//
//   sin2_t = eta^2 (1 - cos_i^2); total internal reflection when > 1
//   r_s = (eta cos_i - cos_t) / (eta cos_i + cos_t)
//   r_p = (cos_i - eta cos_t) / (cos_i + eta cos_t)
//   F   = (r_s^2 + r_p^2) / 2
//   reflected = d + 2 cos_i n_f,  refracted = eta d + (eta cos_i - cos_t) n_f
//
// where n_f is the normal turned to face the arriving ray.
//
// Its BSDF (contracts/bsdf.h) is two delta lobes, reflection and
// transmission, so evaluate and pdf are 0 and only sample() answers: it
// reflects with probability F, value F / |cos|, and refracts with
// probability 1 - F, value (1 - F) / |cos| / eta_t^2, eta_t the ratio of the
// index wi's side to wo's. The 1 / eta_t^2 is radiance's: it is n^2-scaled
// across a boundary (pbrt-v4, DielectricBxDF, TransportMode::Radiance), and
// for a path through a whole sphere it cancels. A sample's weight is 1 when
// it reflects and 1 / eta_t^2 when it refracts.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/materials/dielectric.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

// The Fresnel reflectance F for unpolarized light at a smooth boundary,
// cos_i the cosine of the angle of incidence, in [0, 1], and eta = n_from /
// n_to; and, through `cos_t`, the cosine of the refracted angle. 1 under
// total internal reflection, cos_t then 0. The one Fresnel every smooth
// boundary here uses: the glass's (below) and the coat's (coated.metal.h).
inline float fresnel_reflectance(float cos_i, float eta, thread float& cos_t) {
    const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) {
        cos_t = 0.0f;
        return 1.0f;
    }
    cos_t = metal::sqrt(1.0f - sin2_t);
    const float r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const float r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (r_s * r_s + r_p * r_p);
}

struct Boundary {
    bool entering;         // arriving from outside, against the outward normal
    bool total_internal;   // no refracted direction; reflectance is 1
    float reflectance;     // F
    float3 facing;         // the normal on the side the ray arrives from
    float3 reflected;
    float3 refracted;      // unit length unless total_internal
};

// `direction` and `normal` are unit length.
inline Boundary dielectric_boundary(serenity::materials::DielectricData glass, float3 direction, float3 normal) {
    Boundary b;
    float cos_i = -metal::dot(direction, normal);
    b.entering = cos_i > 0.0f;
    b.facing = b.entering ? normal : -normal;
    cos_i = metal::abs(cos_i);
    const float eta = b.entering ? 1.0f / glass.ior : glass.ior;

    b.reflected = direction + 2.0f * cos_i * b.facing;
    float cos_t;
    b.reflectance = fresnel_reflectance(cos_i, eta, cos_t);
    b.total_internal = eta * eta * (1.0f - cos_i * cos_i) > 1.0f;
    b.refracted = b.total_internal ? float3(0.0f)
                                   : metal::normalize(eta * direction + (eta * cos_i - cos_t) * b.facing);
    return b;
}

// `n` is the outward normal; wo points away from the surface, toward where
// the light goes, so the light arrives along -wo's mirror image or through.
inline serenity::contracts::BsdfSample dielectric_sample(serenity::contracts::Bsdf bsdf, float3 n, float3 wo,
                                                         float choose) {
    serenity::materials::DielectricData glass;
    glass.ior = bsdf.ior;
    // Traced backwards: the camera's ray arrives along -wo.
    const Boundary b = dielectric_boundary(glass, -wo, n);
    const float cos_o = metal::abs(metal::dot(wo, n));
    serenity::contracts::BsdfSample s;
    if (b.total_internal || choose < b.reflectance) {
        s.direction = to_packed(b.reflected);
        s.pdf = b.total_internal ? 1.0f : b.reflectance;
        s.value = to_packed(float3(s.pdf / cos_o));
        s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_delta;
        return s;
    }
    const float cos_t = metal::abs(metal::dot(b.refracted, n));
    // wi's side is the side refracted into: inside when wo is outside.
    const float eta_t = b.entering ? bsdf.ior : 1.0f / bsdf.ior;
    s.direction = to_packed(b.refracted);
    s.pdf = 1.0f - b.reflectance;
    s.value = to_packed(float3(s.pdf / cos_t / (eta_t * eta_t)));
    s.lobe = serenity::contracts::lobe_transmission | serenity::contracts::lobe_delta;
    return s;
}

}  // namespace shaders
}  // namespace serenity
