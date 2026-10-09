#pragma once

// Axis: Pass (display encoding, shared by the passes that write a frame's
// target).
//
// encode_srgb(): a linear color, for display as it is: a color brighter
// than the display shows is scaled down by its largest channel, so it keeps
// its hue (a firefly stays yellow rather than clipping to white); then
// linear to sRGB's transfer function, for an 8-bit target the display reads
// as sRGB. The display pass's (passes/display/display.h). transfer_srgb():
// the transfer function alone, for a color already in [0, 1], as the
// tone-map pass's roll-off leaves it (passes/tone_map/tone_map.h).

#include <metal_stdlib>

namespace serenity {
namespace shaders {

inline float3 encode_srgb(float3 linear) {
    const float largest = metal::max(metal::max(linear.r, linear.g), linear.b);
    const float3 c = metal::saturate(largest > 1.0f ? linear / largest : linear);
    return metal::select(1.055f * metal::pow(c, 1.0f / 2.4f) - 0.055f, 12.92f * c, c <= 0.0031308f);
}

}  // namespace shaders
}  // namespace serenity
