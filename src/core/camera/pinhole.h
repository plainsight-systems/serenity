#pragma once

// Axis: Camera (pinhole).
//
// Framing a camera (contracts/camera.h) as a pinhole, for an image of a
// given size: the pure function from the camera a scene describes to the
// form a shader generates rays from (F.8), and the conditions a camera must
// meet to be framed. The ray formula is in the contract; the shader half is
// metal/camera/pinhole.metal.h.
//
// The camera's pose at t joins this family when the camera moves; until then
// a frame's camera is the scene's.
//
// CPU only: shaders receive the framed contracts::CameraData.

#include "core/contracts/camera.h"
#include "core/frame/extent.h"

namespace serenity::camera {

// Whether `camera` can be framed: the conditions in contracts/camera.h. If
// not, `reason` names the one it fails. The scene reader refuses such a
// camera.
bool valid(const contracts::Camera& camera, const char** reason);

// `camera` framed for an image of `size`. `camera` must be valid and `size`
// non-empty.
contracts::CameraData shader_form(const contracts::Camera& camera, frame::Extent size);

}  // namespace serenity::camera
