#pragma once

// Shared shader mathematics: sRGB's transfer function (IEC 61966-2-1), from
// a linear value in [0, 1] to what an 8-bit target the display reads as
// sRGB holds. The presenting pass ends in it (passes/tone_map/tone_map.h);
// how it brings its colors into [0, 1] first is its own. A mapping, owned by no
// pass. And the luminance of a linear color in its primaries, Rec. 709's, which
// ReSTIR PT's targets weigh by (metal/integrator/nee.metal.h,
// metal/passes/restir_pt/).
//
// Translated line for line from Metal's metal/math/srgb.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// IEC 61966-2-1's constants: below the cutoff the curve is linear, with
// that slope; above it, a power of 1 / gamma, scaled and offset.
inline constexpr float srgb_cutoff = 0.0031308f;
inline constexpr float srgb_linear_slope = 12.92f;
inline constexpr float srgb_gamma = 2.4f;
inline constexpr float srgb_scale = 1.055f;
inline constexpr float srgb_offset = 0.055f;

// Rec. 709's (and sRGB's) luminance weights for a linear R, G, B.
// Three numbers, not Metal's one float3 constant: CUDA's device code takes
// a namespace-scope constant of a scalar type alone.
inline constexpr float rec709_red = 0.2126f;
inline constexpr float rec709_green = 0.7152f;
inline constexpr float rec709_blue = 0.0722f;

// Rec. 709 luminance of a linear sRGB radiance.
__device__ inline float luminance(float3 radiance) {
  return dot(radiance, make_float3(rec709_red, rec709_green, rec709_blue));
}

__device__ inline float3 transfer_srgb(float3 c) {
  return select(srgb_scale * pow(c, 1.0f / srgb_gamma) - srgb_offset,
                srgb_linear_slope * c, c <= srgb_cutoff);
}

}  // namespace shaders
}  // namespace serenity
