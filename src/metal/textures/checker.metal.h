#pragma once

// Axis: Texture (shader half of textures/checker.h).
//
// The checker's color at point p, by the rule in textures/checker.h.

#include <metal_stdlib>

#include "core/textures/checker.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

inline float3 checker(serenity::textures::CheckerData data, float3 point) {
    const float squares = metal::floor(point.x / data.size) + metal::floor(point.z / data.size);
    const bool even = metal::fmod(metal::abs(squares), 2.0f) < 0.5f;
    return even ? to_float3(data.a) : to_float3(data.b);
}

}  // namespace shaders
}  // namespace serenity
