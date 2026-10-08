#pragma once

// Axis: Camera (pinhole).
//
// A pinhole camera: as a scene describes it (where it is, what it looks at,
// which way is up, how wide it sees), and as a shader uses it (an origin and
// three vectors spanning the image), with the pure function from one to the
// other for a given image size (shader_form, F.8).
//
// Ray generation is the shader half (metal/camera/pinhole.metal.h): pixel
// (x, y) of a W x H image, at its center, looks along
//
//   forward + (2 (x + 0.5) / W - 1) right + (1 - 2 (y + 0.5) / H) up
//
// from origin, rows counting down from the top. `right` and `up` are scaled
// so that this spans the field of view: |up| = tan(vertical_fov / 2) and
// |right| = |up| x W / H, so pixels are square whatever the image's shape.

#include "core/contracts/float3.h"

#if !defined(__METAL_VERSION__)
#include "core/frame/extent.h"
#endif

namespace serenity {
namespace camera {

// What a frame's shaders read (contracts/frame_constants.h carries it).
struct PinholeData {
    contracts::Float3 origin;
    float padding0;
    contracts::Float3 forward;  // unit length
    float padding1;
    contracts::Float3 right;
    float padding2;
    contracts::Float3 up;
    float padding3;
};

static_assert(sizeof(PinholeData) == 64, "PinholeData must be the same 64 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// The camera as a scene file describes it.
struct Pinhole {
    contracts::Float3 position;
    contracts::Float3 look_at;     // not equal to position
    contracts::Float3 up;          // not parallel to look_at - position
    float vertical_fov_degrees;    // strictly between 0 and 180
};

// Whether `pinhole` can be framed: the conditions above. If not, `reason`
// names the one it fails. The scene reader refuses such a camera.
bool valid(const Pinhole& pinhole, const char** reason);

// The shader's form of `pinhole` for an image of `size`. `pinhole` must be
// valid and `size` non-empty.
PinholeData shader_form(const Pinhole& pinhole, frame::Extent size);
#endif

}  // namespace camera
}  // namespace serenity
