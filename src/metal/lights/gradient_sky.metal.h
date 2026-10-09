#pragma once

// Axis: Light (shader half of lights/gradient_sky.h).
//
// The radiance arriving from direction d, by the blend lights/gradient_sky.h
// states: smoothstep clamps d.y, so below the horizon it is the horizon.

#include <metal_stdlib>

#include "core/lights/gradient_sky.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

inline float3 gradient_sky(serenity::lights::GradientSkyData sky, float3 direction) {
    const float blend = metal::smoothstep(0.0f, 1.0f, direction.y);
    return to_float3(sky.horizon) + (to_float3(sky.zenith) - to_float3(sky.horizon)) * blend;
}

}  // namespace shaders
}  // namespace serenity
