#pragma once

// The formulas the tests check the code against, worked out once, in double
// (ES.3, F.10): the sRGB encoding to 8 bits and back, the camera's ray
// through a point of the image (contracts/camera.h), and Fresnel's
// reflectance at a smooth boundary (metal/materials/fresnel.metal.h).

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/contracts/camera.h"
#include "core/frame/extent.h"
#include "support/vector.h"

namespace serenity::tests {

// Linear `v` encoded as sRGB, to 8 bits, clamped to [0, 1] first.
inline int srgb8(double v) {
    v = std::clamp(v, 0.0, 1.0);
    const double e = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
    return static_cast<int>(std::lround(e * 255.0));
}

// An 8-bit sRGB value back to linear.
inline double linear_of(int encoded) {
    const double c = encoded / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

// The pinhole camera ray's unit direction through (px, py), in pixels from
// the top left, of an image of `size` (contracts/camera.h).
inline Vec3 camera_direction(const contracts::CameraData& c, double px, double py, frame::Extent size) {
    const double sx = 2.0 * px / size.width - 1.0;
    const double sy = 1.0 - 2.0 * py / size.height;
    return normalized(vec(c.forward) + sx * vec(c.right) + sy * vec(c.up));
}

// A pixel of an image, from its top left.
struct Pixel {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

// The pixel the pinhole camera `c` sees `point` in, of an image of `size`:
// camera_direction() inverted.
inline Pixel pixel_of(const contracts::CameraData& c, Vec3 point, frame::Extent size) {
    const Vec3 d = normalized(point - vec(c.origin));
    const Vec3 on_plane = (1.0 / dot(d, vec(c.forward))) * d;
    const double sx = dot(on_plane, vec(c.right)) / dot(vec(c.right), vec(c.right));
    const double sy = dot(on_plane, vec(c.up)) / dot(vec(c.up), vec(c.up));
    return {static_cast<std::uint32_t>((sx + 1.0) * 0.5 * size.width),
            static_cast<std::uint32_t>((1.0 - sy) * 0.5 * size.height)};
}

// The unpolarized Fresnel reflectance of light arriving at a smooth boundary
// at cos_i, with eta = n_from / n_to; 1 past the critical angle.
inline double fresnel_reflectance(double cos_i, double eta) {
    const double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0) {
        return 1.0;
    }
    const double cos_t = std::sqrt(1.0 - sin2_t);
    const double r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const double r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

// The reflectance from air into a medium of index `ior`.
inline double fresnel_from_air(double cos_i, double ior) {
    return fresnel_reflectance(cos_i, 1.0 / ior);
}

}  // namespace serenity::tests
