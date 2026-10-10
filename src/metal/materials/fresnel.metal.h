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

// What a smooth boundary does to light arriving at it: one result (F.21).
struct Fresnel {
    float reflectance;    // F, for unpolarized light; 1 under total internal reflection
    float cos_t;          // the cosine of the refracted angle; 0 under total internal reflection
    bool total_internal;  // sin^2 of the refracted angle past 1: nothing is refracted
};

// The Fresnel reflectance at a smooth boundary, cos_i the cosine of the
// angle of incidence, in [0, 1], and eta = n_from / n_to.
inline Fresnel fresnel(float cos_i, float eta) {
    const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) {
        return Fresnel{1.0f, 0.0f, true};
    }
    const float cos_t = metal::sqrt(1.0f - sin2_t);
    const float r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const float r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return Fresnel{0.5f * (r_s * r_s + r_p * r_p), cos_t, false};
}

}  // namespace shaders
}  // namespace serenity
