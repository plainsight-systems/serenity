#pragma once

// Axis: Material (shader half of materials/rough.h).
//
// A rough surface: its color at a point, its texture's through the texture
// reference or its constant color; and its BSDF (contracts/bsdf.h), Lambert's,
// albedo / pi on the side wo is on, sampled by the cosine (Malley's method),
// pdf = |cos wi| / pi. One lobe: reflection, diffuse.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/materials/rough.h"
#include "metal/device/layout.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {

inline float3 rough_color(serenity::materials::RoughData rough, Textures textures, float3 point) {
    if (rough.texture.index == serenity::contracts::no_texture) {
        return to_float3(rough.color);
    }
    return evaluate_texture(textures, rough.texture, point);
}

// `n` is the normal turned to wo's side; wi on the other side scatters
// nothing.
inline float3 lambert_evaluate(serenity::contracts::Bsdf bsdf, float3 n, float3 wi) {
    return metal::dot(wi, n) > 0.0f ? to_float3(bsdf.color) * M_1_PI_F : float3(0.0f);
}

inline float lambert_pdf(float3 n, float3 wi) {
    return metal::max(0.0f, metal::dot(wi, n)) * M_1_PI_F;
}

inline serenity::contracts::BsdfSample lambert_sample(serenity::contracts::Bsdf bsdf, float3 n, float2 u) {
    const float3 wi = cosine_direction(n, u);
    serenity::contracts::BsdfSample s;
    s.direction = to_packed(wi);
    s.pdf = lambert_pdf(n, wi);
    s.value = to_packed(lambert_evaluate(bsdf, n, wi));
    s.lobe = serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse;
    return s;
}

}  // namespace shaders
}  // namespace serenity
