#pragma once

// Axis: Texture (shader half of textures/wood.h).
//
// The plank tabletop's color at point p, by core/textures/wood.h's steps.

#include <metal_stdlib>

#include "core/textures/wood.h"
#include "metal/device/layout.metal.h"
#include "metal/math/hash.metal.h"
#include "metal/textures/noise.metal.h"

namespace serenity {
namespace shaders {

inline float3 wood(serenity::textures::WoodData data, float3 p) {
    // Step 1: the board, and where across it p lies.
    const float across = p.z / data.board;
    const float b = metal::floor(across);
    const float u = across - b;
    // Step 2: the board's log. b as a 32-bit integer: within its range, as
    // core/textures/wood.h bounds p.z / board (at most 10^9; ES.46).
    const uint3 h = pcg3d(uint3(as_type<uint>(int(b)), data.seed, serenity::textures::wood_salt));
    const float3 unit = unit_float3(h);  // the top 24 bits of each, over 2^24
    const float depth = (serenity::textures::wood_depth_least +
                         unit.y * (serenity::textures::wood_depth_most - serenity::textures::wood_depth_least)) *
                        data.board;
    const float shade = 1.0f + serenity::textures::wood_board_shade * (2.0f * unit.z - 1.0f);
    // Step 3: the ring radius, and step 4: the grain's waver, at height 0.
    const float a = (u - unit.x) * data.board;
    const float3 q =
        float3(p.x / serenity::textures::wood_waver_along, 0.0f, p.z / serenity::textures::wood_waver_across);
    const float r =
        metal::sqrt(a * a + depth * depth) + serenity::textures::wood_waver * data.ring * fbm(q, 3u, data.seed);
    // Step 5: the growth ring, light to dark.
    const float t = metal::fract(r / data.ring);
    const float w = metal::smoothstep(serenity::textures::wood_latewood, 1.0f, t);
    float3 color = metal::mix(to_float3(data.light), to_float3(data.dark), w);
    // Step 6: the pores.
    const float3 g =
        float3(p.x / serenity::textures::wood_pores_along, 0.0f, p.z / serenity::textures::wood_pores_across);
    color *= 1.0f - serenity::textures::wood_pores * metal::saturate(0.5f + 0.5f * gradient_noise(g, data.seed + 1u));
    // Step 7: the board's shade and the seam.
    color *= shade;
    if (metal::min(u, 1.0f - u) * data.board < serenity::textures::wood_seam) {
        color *= serenity::textures::wood_seam_shade;
    }
    return color;
}

}  // namespace shaders
}  // namespace serenity
