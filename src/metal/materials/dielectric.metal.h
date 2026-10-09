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

#include <metal_stdlib>

#include "core/materials/dielectric.h"

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
    const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    b.total_internal = sin2_t > 1.0f;
    if (b.total_internal) {
        b.reflectance = 1.0f;
        b.refracted = float3(0.0f);
        return b;
    }
    const float cos_t = metal::sqrt(1.0f - sin2_t);
    const float r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const float r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    b.reflectance = 0.5f * (r_s * r_s + r_p * r_p);
    b.refracted = metal::normalize(eta * direction + (eta * cos_i - cos_t) * b.facing);
    return b;
}

}  // namespace shaders
}  // namespace serenity
