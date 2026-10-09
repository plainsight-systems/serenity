#pragma once

// Axis: Pass (display), its shader half (display.h).
//
// A linear color, shown as it is: a color brighter than the display shows is
// scaled down by its largest channel, so it keeps its hue (a firefly stays
// yellow rather than clipping to white), then encoded with sRGB's transfer
// function (metal/math/srgb.metal.h). The rule the preview and path passes
// applied to their own output before the radiance image came between them
// and the target, unchanged, so a graph of a light pass and the display
// pass shows what that light pass alone showed before, byte for byte.

#include <metal_stdlib>

#include "metal/math/srgb.metal.h"

namespace serenity {
namespace shaders {

inline float3 display_color(float3 linear) {
    const float largest = metal::max(metal::max(linear.r, linear.g), linear.b);
    return transfer_srgb(metal::saturate(largest > 1.0f ? linear / largest : linear));
}

}  // namespace shaders
}  // namespace serenity
