#pragma once

// Shared shader mathematics: sRGB's transfer function (IEC 61966-2-1), from
// a linear value in [0, 1] to what an 8-bit target the display reads as
// sRGB holds. The presenting pass ends in it (passes/tone_map/tone_map.h);
// how it brings its colors into [0, 1] first is its own. A mapping, owned by no pass. And the luminance of a
// linear color in its primaries, Rec. 709's, which ReSTIR PT's targets
// weigh by (metal/integrator/nee.metal.h, metal/passes/restir_pt/).

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

// Rec. 709's (and sRGB's) luminance weights for a linear R, G, B.
constant constexpr float3 rec709_luminance = float3(0.2126f, 0.7152f, 0.0722f);

// Rec. 709 luminance of a linear sRGB radiance.
inline float luminance(float3 radiance) {
    return metal::dot(radiance, rec709_luminance);
}

inline float3 transfer_srgb(float3 c) {
    return metal::select(srgb_scale * metal::pow(c, 1.0f / srgb_gamma) - srgb_offset, srgb_linear_slope * c,
                         c <= srgb_cutoff);
}

}  // namespace shaders
}  // namespace serenity
