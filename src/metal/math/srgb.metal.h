#pragma once

// Shared shader mathematics: sRGB's transfer function (IEC 61966-2-1), from
// a linear value in [0, 1] to what an 8-bit target the display reads as
// sRGB holds. Every presenting pass ends in it (passes/display/display.h,
// passes/tone_map/tone_map.h); how a pass brings its colors into [0, 1]
// first is its own. A mapping, owned by no pass.

#include <metal_stdlib>

namespace serenity {
namespace shaders {

inline float3 transfer_srgb(float3 c) {
    return metal::select(1.055f * metal::pow(c, 1.0f / 2.4f) - 0.055f, 12.92f * c, c <= 0.0031308f);
}

}  // namespace shaders
}  // namespace serenity
