#pragma once

// Vectors in double, for the tests' references worked out on the CPU (ES.3,
// F.10): the precision a reference needs, apart from the float the code
// under test computes in.

#include <cmath>

#include "core/contracts/float3.h"

namespace serenity::tests {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

constexpr Vec3 operator+(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

constexpr Vec3 operator-(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

constexpr Vec3 operator-(Vec3 a) noexcept {
    return {-a.x, -a.y, -a.z};
}

constexpr Vec3 operator*(double s, Vec3 a) noexcept {
    return {s * a.x, s * a.y, s * a.z};
}

constexpr double dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(Vec3 a) noexcept {
    return std::sqrt(dot(a, a));
}

inline Vec3 normalized(Vec3 a) noexcept {
    return (1.0 / length(a)) * a;
}

// The unit vector along (x, y, z).
inline Vec3 unit(double x, double y, double z) noexcept {
    return normalized({x, y, z});
}

constexpr Vec3 vec(contracts::Float3 f) noexcept {
    return {f.x, f.y, f.z};
}

}  // namespace serenity::tests
