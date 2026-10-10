#pragma once

// Axis: Texture (shader half of textures/swirl.h).
//
// A marble core's twisting vanes at q, a point in the core's own
// coordinates, by core/textures/swirl.h's steps. No branch (GPU.4).

#include <metal_stdlib>

#include "core/textures/swirl.h"
#include "metal/device/layout.metal.h"
#include "metal/textures/noise.metal.h"

namespace serenity {
namespace shaders {

inline float3 swirl(serenity::textures::SwirlData data, float3 q) {
    // Step 1: q's angle about the axis, y, and its height.
    const float phi = metal::atan2(q.z, q.x);
    // Step 2: its phase among the vanes, the noise read on the unit
    // cylinder at q's angle and height.
    const float3 on_cylinder = float3(metal::cos(phi), q.y, metal::sin(phi));
    const float s = float(data.vanes) * (phi * (0.5f * M_1_PI_F) + data.twist * q.y) +
                    serenity::textures::swirl_waver * fbm(serenity::textures::swirl_scale * on_cylinder,
                                                          serenity::textures::swirl_waver_octaves, data.seed);
    // Step 3: its color, a's bands and b's, their edges soft.
    const float v = metal::abs(2.0f * metal::fract(s) - 1.0f);
    const float w =
        metal::smoothstep(0.5f - serenity::textures::swirl_edge, 0.5f + serenity::textures::swirl_edge, v);
    return metal::mix(to_float3(data.a), to_float3(data.b), w);
}

}  // namespace shaders
}  // namespace serenity
