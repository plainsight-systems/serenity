#pragma once

// Axis: Texture (shader half of textures/wood.h).
//
// The plank tabletop's color at point p, by core/textures/wood.h's steps.

#include <metal_stdlib>

#include "core/textures/wood.h"
#include "metal/device/layout.metal.h"
#include "metal/textures/noise.metal.h"

namespace serenity {
namespace shaders {

inline float3 wood(serenity::textures::WoodData data, float3 p) {
    namespace tx = serenity::textures;
    // Step 1: the board, and where across it p lies.
    const float across = p.z / data.board;
    const float b = metal::floor(across);
    const float u = across - b;
    // Step 2: the board's log.
    const uint3 h = pcg3d(uint3(as_type<uint>(int(b)), data.seed, tx::wood_salt));
    const float3 unit = float3(h >> 8u) / 16777216.0f;
    const float depth = (tx::wood_depth_least + unit.y * (tx::wood_depth_most - tx::wood_depth_least)) * data.board;
    const float shade = 1.0f + tx::wood_board_shade * (2.0f * unit.z - 1.0f);
    // Step 3: the ring radius.
    const float a = (u - unit.x) * data.board;
    float r = metal::sqrt(a * a + depth * depth);
    // Step 4: the grain's waver, at height 0.
    const float3 q = float3(p.x / tx::wood_waver_along, 0.0f, p.z / tx::wood_waver_across);
    r += tx::wood_waver * data.ring * fbm(q, 3u, data.seed);
    // Step 5: the growth ring, light to dark.
    const float t = metal::fract(r / data.ring);
    const float w = metal::smoothstep(tx::wood_latewood, 1.0f, t);
    float3 color = metal::mix(to_float3(data.light), to_float3(data.dark), w);
    // Step 6: the pores.
    const float3 g = float3(p.x / tx::wood_pores_along, 0.0f, p.z / tx::wood_pores_across);
    color *= 1.0f - tx::wood_pores * metal::saturate(0.5f + 0.5f * gradient_noise(g, data.seed + 1u));
    // Step 7: the board's shade and the seam.
    color *= shade;
    if (metal::min(u, 1.0f - u) * data.board < tx::wood_seam) {
        color *= tx::wood_seam_shade;
    }
    return color;
}

}  // namespace shaders
}  // namespace serenity
