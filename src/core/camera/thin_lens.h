#pragma once

// Axis: Camera (thin lens).
//
// Framing a camera (contracts/camera.h) as a thin lens, a pinhole when its
// radius is 0, for an image of a given size: the pure function from the
// camera a scene describes to the form a shader generates rays from (F.8),
// and the conditions a camera must meet to be framed. The ray formula is in
// the contract; the shader half is metal/camera/thin_lens.metal.h.
//
// The camera's pose at t joins this family when the camera moves; until then
// a frame's camera is the scene's.
//
// The framing is computed in double and narrowed to float once, at the end:
// for any finite camera the difference of two floats, its length and its
// cross products are finite in double, so the conditions invalid() checks
// are the whole of what framing needs, and a camera they accept frames to
// finite numbers, never to a NaN (I.7). Each condition is written to pass
// only for the numbers it accepts, so a NaN fails it.
//
// CPU only: shaders receive the framed contracts::CameraData.

#include <optional>
#include <string_view>

#include "core/contracts/camera.h"
#include "core/frame/extent.h"

namespace serenity::camera {

// Whether `camera` can be framed: none if it can, the condition of
// contracts/camera.h it fails if not, as the frame graph's invalid() says
// why a schedule is refused (F.20: a returned value, not an out-parameter).
// The scene reader refuses such a camera.
std::optional<std::string_view> invalid(const contracts::Camera& camera);

// `camera` framed for an image of `size`. Preconditions, checked (I.5, E.2):
// `camera` valid (invalid() gives none) and `size` non-empty; throws
// std::invalid_argument otherwise.
contracts::CameraData shader_form(const contracts::Camera& camera,
                                  frame::Extent size);

// The circle of confusion's scale for `camera` in an image of `size`
// (contract 21, core/contracts/depth_of_field.h): a point at view depth z
// blurs through the lens to a disc of radius scale |z - F| / z pixels, F
// the focus distance. The rays through the lens's points meet at the point
// where the pinhole ray meets the plane of focus (contracts/camera.h), so
// the rays toward a point at depth z cross that plane in a disc of radius
// r |z - F| / z (similar triangles, apex at the point, base the lens), and
// the image shows a length L on the plane of focus as L H / (2 F
// tan(fov / 2)) pixels; so
//
//   scale = r H / (2 F tan(fov / 2)),
//
// in double, narrowed once, as the framing (F.8). 0 for a pinhole. The
// depth-of-field pass blurs by it; tests/gpu/camera_test.cpp holds it to
// the lens rays camera/thin_lens.metal.h traces (a test of a match between
// implementations). Precondition as shader_form()'s, checked.
float circle_of_confusion_scale(const contracts::Camera& camera,
                                frame::Extent size);

// The camera's view axes, each unit length: right, up, and backward
// (toward the viewer, -forward), the axes the denoiser's network reads
// normals in (core/contracts/denoise.h; training/regions.py, view_axes,
// the same construction). Precondition as shader_form()'s, checked. Each
// member has a default member initializer, as a Camera's does (contract
// 8), so a ViewAxes is never garbage (ES.20).
struct ViewAxes {
  contracts::Float3 right{};
  contracts::Float3 up{};
  contracts::Float3 backward{};
};
ViewAxes view_axes(const contracts::Camera& camera);

}  // namespace serenity::camera
