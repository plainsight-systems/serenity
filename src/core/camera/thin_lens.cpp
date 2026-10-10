#include "core/camera/thin_lens.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace serenity::camera {

namespace {

using contracts::Float3;

// The framing's arithmetic, in double (thin_lens.h says why).
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

constexpr Vec3 widened(Float3 a) noexcept {
    return {a.x, a.y, a.z};
}

Float3 narrowed(Vec3 a) noexcept {
    return {static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)};
}

constexpr Vec3 minus(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

constexpr Vec3 scaled(Vec3 a, double s) noexcept {
    return {a.x * s, a.y * s, a.z * s};
}

constexpr Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double length(Vec3 a) noexcept {
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

Vec3 normalized(Vec3 a) noexcept {
    return scaled(a, 1.0 / length(a));
}

bool finite(Float3 a) noexcept {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

// Below this, a length or the sine of an angle is taken as zero: far below
// any scene's scale, and far above double rounding at it.
constexpr double tiny = 1e-6;

// Half a field of view's angle in radians, per degree of the whole.
constexpr double half_radians_per_degree = std::numbers::pi / 360.0;

}  // namespace

std::optional<std::string_view> invalid(const contracts::Camera& camera) {
    if (!finite(camera.position) || !finite(camera.look_at) || !finite(camera.up) ||
        !std::isfinite(camera.vertical_fov_degrees)) {
        return "a camera value is not a finite number";
    }
    // Each test passes only for a number at least tiny, so a NaN fails it
    // (the form the lens's tests take too).
    const Vec3 view = minus(widened(camera.look_at), widened(camera.position));
    if (!(length(view) >= tiny)) {
        return "look_at is the camera's position";
    }
    if (!(length(widened(camera.up)) >= tiny)) {
        return "up has no length";
    }
    if (!(length(cross(normalized(view), normalized(widened(camera.up)))) >= tiny)) {
        return "up is parallel to the direction the camera looks";
    }
    if (!(camera.vertical_fov_degrees > 0.0f && camera.vertical_fov_degrees < 180.0f)) {
        return "vertical_fov_degrees is not strictly between 0 and 180";
    }
    if (!(std::isfinite(camera.lens_radius) && camera.lens_radius >= 0.0f)) {
        return "the lens's radius is not a finite number, 0 or more";
    }
    if (!(std::isfinite(camera.focus_distance) && camera.focus_distance > 0.0f)) {
        return "the lens's focus is not a finite number greater than 0";
    }
    return std::nullopt;
}

contracts::CameraData shader_form(const contracts::Camera& camera, frame::Extent size) {
    if (size.width == 0 || size.height == 0) {
        throw std::invalid_argument("shader_form: an image of no pixels cannot be framed");
    }
    if (const std::optional<std::string_view> reason = invalid(camera)) {
        throw std::invalid_argument("shader_form: the camera cannot be framed: " + std::string(*reason));
    }
    const Vec3 forward = normalized(minus(widened(camera.look_at), widened(camera.position)));
    const Vec3 right_unit = normalized(cross(forward, widened(camera.up)));
    const Vec3 up_unit = cross(right_unit, forward);  // unit: the two are orthonormal

    const double half_height = std::tan(static_cast<double>(camera.vertical_fov_degrees) * half_radians_per_degree);
    const double half_width = half_height * static_cast<double>(size.width) / static_cast<double>(size.height);

    contracts::CameraData data{};
    data.origin = camera.position;
    data.forward = narrowed(forward);
    data.right = narrowed(scaled(right_unit, half_width));
    data.up = narrowed(scaled(up_unit, half_height));
    data.lens_radius = camera.lens_radius;
    data.focus_distance = camera.focus_distance;
    return data;
}

}  // namespace serenity::camera
