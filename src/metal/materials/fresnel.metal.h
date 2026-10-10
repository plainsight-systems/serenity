#pragma once

// Axis: Material (shared by the kinds with a smooth boundary).
//
// The Fresnel equations for a smooth boundary between two dielectrics, for
// unpolarized light: what the glass (dielectric.metal.h) and the clear coat
// (coated.metal.h) each reflect at their surface. Owned by no one kind, so
// either changes without the other, and a new smooth-boundary kind depends
// on this, not on a kind. The CPU's double copy for load-time integrals is
// core/materials/coated.cpp's.

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// The Fresnel reflectance F for unpolarized light at a smooth boundary,
// cos_i the cosine of the angle of incidence, in [0, 1], and eta = n_from /
// n_to; and, through `cos_t`, the cosine of the refracted angle. 1 under
// total internal reflection, cos_t then 0.
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

}  // namespace shaders
}  // namespace serenity
