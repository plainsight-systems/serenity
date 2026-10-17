#pragma once

// Vectors in double, for the tests' references worked out on the CPU (ES.3,
// F.10): the precision a reference needs, apart from the float the code
// under test computes in; and the float vectors the code under test reads
// and writes, made from them and compared.

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

constexpr Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

constexpr Vec3 vec(contracts::Float3 f) noexcept {
    return {f.x, f.y, f.z};
}

// `v` narrowed to the float the GPU's records hold.
constexpr contracts::Float3 float3(Vec3 v) noexcept {
    return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
}

// Whether two float vectors are the same, component by component: exact,
// for the results that must repeat to the bit.
constexpr bool same(contracts::Float3 a, contracts::Float3 b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

}  // namespace serenity::tests
