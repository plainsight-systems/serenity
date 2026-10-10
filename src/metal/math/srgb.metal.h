#pragma once

// Shared shader mathematics: sRGB's transfer function (IEC 61966-2-1), from
// a linear value in [0, 1] to what an 8-bit target the display reads as
// sRGB holds. Every presenting pass ends in it (passes/display/display.h,
// passes/tone_map/tone_map.h); how a pass brings its colors into [0, 1]
// first is its own. A mapping, owned by no pass.

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// IEC 61966-2-1's constants: below the cutoff the curve is linear, with
// that slope; above it, a power of 1 / gamma, scaled and offset.
constant constexpr float srgb_cutoff = 0.0031308f;
constant constexpr float srgb_linear_slope = 12.92f;
constant constexpr float srgb_gamma = 2.4f;
constant constexpr float srgb_scale = 1.055f;
constant constexpr float srgb_offset = 0.055f;

inline float3 transfer_srgb(float3 c) {
    return metal::select(srgb_scale * metal::pow(c, 1.0f / srgb_gamma) - srgb_offset, srgb_linear_slope * c,
                         c <= srgb_cutoff);
}

}  // namespace shaders
}  // namespace serenity
