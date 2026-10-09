#pragma once

// Contract 8: the camera. Owned by Camera; carried by the Frame graph as a
// frame input (frame/frame_inputs.h) and read by every pass that generates
// rays.
//
// Two forms, for the two sides of the frame:
//
//   - Camera, on the CPU: where the camera is, what it looks at, which way is
//     up and how wide it sees. A scene describes one (scene/scene.h); a frame
//     is rendered through one, its pose at the frame's time (principle 1).
//   - CameraData, shared with shaders: the camera framed for one image, an
//     origin and three vectors spanning it. camera/ frames one into the other
//     for an image's size (camera/pinhole.h).
//
// One form for every camera, not a kind and an index: a frame has exactly one
// camera, and what varies between cameras is parameters, not code. Falcor
// does the same: one CameraData for its pinhole and its thin lens, where a
// pinhole is a lens of no aperture. When depth of field comes, the lens's
// aperture and focus distance join both forms; that is a change to this
// contract, made on purpose.
//
// Ray generation is the shader half (metal/camera/pinhole.metal.h): pixel
// (x, y) of a W x H image, at its center, looks along
//
//   forward + (2 (x + 0.5) / W - 1) right + (1 - 2 (y + 0.5) / H) up
//
// from origin, rows counting down from the top. `right` and `up` are scaled
// so that this spans the field of view: |up| = tan(vertical_fov / 2) and
// |right| = |up| x W / H, so pixels are square whatever the image's shape.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct CameraData {
    Float3 origin;
    float padding0;
    Float3 forward;  // unit length
    float padding1;
    Float3 right;
    float padding2;
    Float3 up;
    float padding3;
};

static_assert(sizeof(CameraData) == 64, "CameraData must be the same 64 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
struct Camera {
    Float3 position;
    Float3 look_at;              // not equal to position
    Float3 up;                   // not parallel to look_at - position
    float vertical_fov_degrees;  // strictly between 0 and 180
};
#endif

}  // namespace contracts
}  // namespace serenity
