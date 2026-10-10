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
// it reflects and 1 / eta_t^2 when it refracts. A wo in the surface's plane,
// |cos| 0, has no sample (pdf 0): its value would divide by 0, and whatever
// arrives along it is weighed by that 0.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/materials/dielectric.h"
#include "metal/device/layout.metal.h"
#include "metal/materials/fresnel.metal.h"

namespace serenity {
namespace shaders {

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

// eta_t of a refraction through glass of index `ior`: the index of wi's
// side over wo's, wi's side being the one refracted into, inside when wo
// is outside (`entering`).
inline float transmitted_eta(float ior, bool entering) {
    return entering ? ior : 1.0f / ior;
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
    if (cos_o <= 0.0f) {
        // wo in the surface's plane: no sample (I.5), not a value of F / 0.
        return serenity::contracts::BsdfSample{};
    }
    serenity::contracts::BsdfSample s;
    if (b.total_internal || choose < b.reflectance) {
        s.direction = to_packed(b.reflected);
        s.pdf = b.total_internal ? 1.0f : b.reflectance;
        s.value = to_packed(float3(s.pdf / cos_o));
        s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_delta;
        return s;
    }
    const float cos_t = metal::abs(metal::dot(b.refracted, n));
    const float eta_t = transmitted_eta(bsdf.ior, b.entering);
    s.direction = to_packed(b.refracted);
    s.pdf = 1.0f - b.reflectance;
    s.value = to_packed(float3(s.pdf / cos_t / (eta_t * eta_t)));
    s.lobe = serenity::contracts::lobe_transmission | serenity::contracts::lobe_delta;
    return s;
}

}  // namespace shaders
}  // namespace serenity
