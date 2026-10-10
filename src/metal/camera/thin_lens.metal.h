#pragma once

// Axis: Camera (shader half of camera/thin_lens.h).
//
// The ray from the camera through a point of a W x H image, by the formula
// in contracts/camera.h: from a point on the lens toward where the pinhole
// ray meets the plane of focus.

#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "metal/device/layout.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

struct CameraRay {
    float3 origin;
    float3 direction;  // unit length
};

// Whether the camera has a lens, rather than a pinhole: the one test every
// caller makes (the path pass draws a lens point only through one).
inline bool has_lens(constant serenity::contracts::CameraData& camera) {
    return camera.lens_radius > 0.0f;
}

// `position` is in pixels from the top-left corner of an image of `size`
// pixels: pixel (x, y)'s center is (x + 0.5, y + 0.5). `lens` is a point in
// the unit square, mapped to the lens's disk (warp.metal.h,
// concentric_disk); unread for a pinhole, whose ray is the pinhole's
// exactly.
inline CameraRay camera_ray(constant serenity::contracts::CameraData& camera, float2 position, float2 lens,
                            uint2 size) {
    const float sx = 2.0f * position.x / float(size.x) - 1.0f;
    const float sy = 1.0f - 2.0f * position.y / float(size.y);
    const float3 forward = to_float3(camera.forward);
    const float3 right = to_float3(camera.right);
    const float3 up = to_float3(camera.up);
    const float3 d = forward + sx * right + sy * up;
    const float3 origin = to_float3(camera.origin);
    if (!has_lens(camera)) {
        return CameraRay{origin, metal::normalize(d)};
    }
    const float3 focus = origin + (camera.focus_distance / metal::dot(d, forward)) * d;
    const float2 on_disk = concentric_disk(lens) * camera.lens_radius;
    const float3 start = origin + on_disk.x * metal::normalize(right) + on_disk.y * metal::normalize(up);
    return CameraRay{start, metal::normalize(focus - start)};
}

}  // namespace shaders
}  // namespace serenity
