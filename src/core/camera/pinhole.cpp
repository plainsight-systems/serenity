#include "core/camera/pinhole.h"

#include <cmath>
#include <numbers>

namespace serenity::camera {

namespace {

using contracts::Float3;

Float3 minus(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 scaled(Float3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

Float3 cross(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float length(Float3 a) {
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

Float3 normalized(Float3 a) {
    return scaled(a, 1.0f / length(a));
}

bool finite(Float3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

// Below this, a length or the sine of an angle is taken as zero: far below
// any scene's scale, and far above float rounding at it.
constexpr float tiny = 1e-6f;

}  // namespace

bool valid(const contracts::Camera& camera, const char** reason) {
    const char* failed = nullptr;
    const Float3 view = minus(camera.look_at, camera.position);
    if (!finite(camera.position) || !finite(camera.look_at) || !finite(camera.up) ||
        !std::isfinite(camera.vertical_fov_degrees)) {
        failed = "a camera value is not a finite number";
    } else if (length(view) < tiny) {
        failed = "look_at is the camera's position";
    } else if (length(camera.up) < tiny) {
        failed = "up has no length";
    } else if (length(cross(normalized(view), normalized(camera.up))) < tiny) {
        failed = "up is parallel to the direction the camera looks";
    } else if (!(camera.vertical_fov_degrees > 0.0f && camera.vertical_fov_degrees < 180.0f)) {
        failed = "vertical_fov_degrees is not strictly between 0 and 180";
    }
    if (reason != nullptr) {
        *reason = failed;
    }
    return failed == nullptr;
}

contracts::CameraData shader_form(const contracts::Camera& camera, frame::Extent size) {
    const Float3 forward = normalized(minus(camera.look_at, camera.position));
    const Float3 right_unit = normalized(cross(forward, camera.up));
    const Float3 up_unit = cross(right_unit, forward);  // unit: the two are orthonormal

    const float half_height = std::tan(camera.vertical_fov_degrees * std::numbers::pi_v<float> / 360.0f);
    const float half_width = half_height * static_cast<float>(size.width) / static_cast<float>(size.height);

    contracts::CameraData data{};
    data.origin = camera.position;
    data.forward = forward;
    data.right = scaled(right_unit, half_width);
    data.up = scaled(up_unit, half_height);
    return data;
}

}  // namespace serenity::camera
