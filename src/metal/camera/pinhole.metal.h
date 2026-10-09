#pragma once

// Axis: Camera (shader half of camera/pinhole.h).
//
// The direction from the camera through a point of a W x H image, by the
// formula in contracts/camera.h, normalized.

#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

// `position` is in pixels from the image's top-left corner: pixel (x, y)'s
// center is (x + 0.5, y + 0.5).
inline float3 pinhole_direction(constant serenity::contracts::CameraData& camera, float2 position, uint width,
                                uint height) {
    const float sx = 2.0f * position.x / float(width) - 1.0f;
    const float sy = 1.0f - 2.0f * position.y / float(height);
    return metal::normalize(to_float3(camera.forward) + sx * to_float3(camera.right) + sy * to_float3(camera.up));
}

}  // namespace shaders
}  // namespace serenity
