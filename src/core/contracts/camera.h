#pragma once

// Contract 8: the camera. Owned by Camera; carried by the Frame graph as a
// frame input (frame/frame_inputs.h) and read by every pass that generates
// rays.
//
// Two forms, for the two sides of the frame:
//
//   - Camera, on the CPU: where the camera is, what it looks at, which way is
//     up, how wide it sees, and its lens. A scene describes one
//     (scene/scene.h); a frame is rendered through one, its pose at the
//     frame's time (principle 1).
//   - CameraData, shared with shaders: the camera framed for one image, an
//     origin and three vectors spanning it, and its lens. camera/ frames one
//     into the other for an image's size (camera/thin_lens.h).
//
// One form for every camera, not a kind and an index: a frame has exactly one
// camera, and what varies between cameras is parameters, not code. Falcor
// does the same: one CameraData for its pinhole and its thin lens, where a
// pinhole is a lens of no aperture. So it is here: a thin lens, of radius
// lens_radius, focused at focus_distance, and a pinhole when the radius is 0.
// What the lens gives is depth of field: what lies on the plane of focus is
// sharp, and the nearer or farther a thing is from it, the larger the disc
// it spreads into, as a camera's lens does; a firefly far behind the marbles
// becomes a soft disc of light (bokeh), which is how its distance reads.
//
// Ray generation is the shader half (metal/camera/thin_lens.metal.h). Point
// (px, py) of a W x H image, in pixels from its top-left corner, is seen
// along the pinhole direction
//
//   d = forward + (2 px / W - 1) right + (1 - 2 py / H) up
//
// from origin, rows counting down from the top; pixel (x, y)'s center is
// (x + 0.5, y + 0.5). `right` and `up` are scaled so that this spans the
// field of view: |up| = tan(vertical_fov / 2) and |right| = |up| x W / H, so
// pixels are square whatever the image's shape. Through the lens, with a
// point u drawn on the unit disk:
//
//   focus = origin + (focus_distance / dot(d, forward)) d, where the pinhole
//           ray meets the plane of focus, square to forward at
//           focus_distance;
//   start = origin + lens_radius (u.x right / |right| + u.y up / |up|), a
//           point on the lens;
//   the ray runs from start toward focus.
//
// So every ray through a point of the image, wherever on the lens it starts,
// meets the same point of the plane of focus (Kolb, Mitchell and Hanrahan
// 1995; pbrt-v4's PerspectiveCamera). With lens_radius 0 the ray is the
// pinhole's, exactly: start is origin and the direction is d's, so a camera
// without a lens renders as it did before lenses. Which of the two a ray
// takes is the frame's camera's, the same for every thread of a dispatch,
// so the branch costs no divergence (GPU.4). Every member of a Camera has a
// default member initializer (C.48, ES.20): a Camera without a lens is a
// pinhole, and a Camera given nothing else is not valid (its look_at is its
// position, camera/thin_lens.h), so none is framed by accident.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct CameraData {
    Float3 origin;
    float lens_radius;     // meters; 0 for a pinhole
    Float3 forward;        // unit length
    float focus_distance;  // meters along forward to the plane of focus; > 0
    Float3 right;
    float padding2;
    Float3 up;
    float padding3;
};

static_assert(sizeof(CameraData) == 64, "CameraData must be the same 64 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<CameraData>, "CameraData is written to the GPU as bytes");
#endif

#if !defined(__METAL_VERSION__)
struct Camera {
    Float3 position{};
    Float3 look_at{};             // not equal to position
    Float3 up{};                  // not parallel to look_at - position
    float vertical_fov_degrees = 0.0f;  // strictly between 0 and 180
    float lens_radius = 0.0f;     // meters, 0 or more; 0 is a pinhole
    float focus_distance = 1.0f;  // meters, greater than 0; read only through a lens
};
#endif

}  // namespace contracts
}  // namespace serenity
