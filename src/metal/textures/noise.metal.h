#pragma once

// Axis: Texture (shader half of textures/noise.h).
//
// Gradient noise and its octaves, by core/textures/noise.h's steps.

#include <metal_stdlib>

#include "core/textures/noise.h"
#include "metal/math/hash.metal.h"

namespace serenity {
namespace shaders {

// noise.h, step 3: as many gradients as its table holds.
constant constexpr uint gradient_count =
    sizeof(serenity::textures::noise_gradients) / sizeof(serenity::textures::noise_gradients[0]);

// Step 3: corner c's gradient, dotted with the offset to the point; step
// 2's hash is pcg3d (math/hash.metal.h).
inline float corner_value(int3 c, float3 offset, uint seed) {
    const uint3 h = pcg3d(as_type<uint3>(c) + uint3(seed));
    const constant int* g = serenity::textures::noise_gradients[h.x % gradient_count];
    return float(g[0]) * offset.x + float(g[1]) * offset.y + float(g[2]) * offset.z;
}

inline float gradient_noise(float3 p, uint seed) {
    // Step 1.
    const float3 cell = metal::floor(p);
    const int3 i = int3(cell);
    const float3 f = p - cell;
    // Steps 2 and 3, at the eight corners.
    const float n000 = corner_value(i, f, seed);
    const float n100 = corner_value(i + int3(1, 0, 0), f - float3(1, 0, 0), seed);
    const float n010 = corner_value(i + int3(0, 1, 0), f - float3(0, 1, 0), seed);
    const float n110 = corner_value(i + int3(1, 1, 0), f - float3(1, 1, 0), seed);
    const float n001 = corner_value(i + int3(0, 0, 1), f - float3(0, 0, 1), seed);
    const float n101 = corner_value(i + int3(1, 0, 1), f - float3(1, 0, 1), seed);
    const float n011 = corner_value(i + int3(0, 1, 1), f - float3(0, 1, 1), seed);
    const float n111 = corner_value(i + int3(1, 1, 1), f - float3(1, 1, 1), seed);
    // Step 4: the quintic fade.
    const float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    const float x00 = metal::mix(n000, n100, u.x), x10 = metal::mix(n010, n110, u.x);
    const float x01 = metal::mix(n001, n101, u.x), x11 = metal::mix(n011, n111, u.x);
    return metal::mix(metal::mix(x00, x10, u.y), metal::mix(x01, x11, u.y), u.z);
}

inline float fbm(float3 p, uint octaves, uint seed) {
    float sum = 0.0f;
    float scale = 1.0f;
    float strength = 1.0f;
    for (uint k = 0; k < octaves; ++k) {
        sum += strength * gradient_noise(p * scale, seed + k);
        scale *= 2.0f;
        strength *= 0.5f;
    }
    return sum;
}

}  // namespace shaders
}  // namespace serenity
