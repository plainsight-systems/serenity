#include "core/shapes/shapes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace serenity::shapes {

namespace {

// A transform's translation and uniform scale, in double, refusing one with
// a rotation: the world-space tests below rely on a placed box staying
// axis-aligned, which the scene format guarantees (contracts/transform.h).
struct Placement {
    std::array<double, 3> t{};
    double scale = 0.0;
};

Placement placement(const contracts::Transform& transform) {
    const auto& m = transform.m;
    if (m[0][1] != 0.0f || m[0][2] != 0.0f || m[1][0] != 0.0f || m[1][2] != 0.0f || m[2][0] != 0.0f ||
        m[2][1] != 0.0f || m[0][0] != m[1][1] || m[1][1] != m[2][2] || !(m[0][0] > 0.0f)) {
        throw std::invalid_argument("a shape's transform has a rotation; the world-space tests take none");
    }
    return Placement{{m[0][3], m[1][3], m[2][3]}, m[0][0]};
}

// `x` rounded to a float no greater, and no less (F.10).
float round_down(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) > x ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
}

float round_up(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) < x ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
}

[[noreturn]] void no_case(const char* where) {
    // After a switch over every ShapeKind: reached only by a value no
    // enumerator names, a corrupt record, which is refused rather than
    // answered as if it were a shape (P.6). "On the surface" or "touches
    // nothing" would weaken every clearance check that asked.
    throw std::logic_error(std::string(where) + ": a shape record whose kind is no ShapeKind");
}

using PlacedSphere = StillShapes::PlacedSphere;
using PlacedBox = StillShapes::PlacedBox;

// Each kind placed, and each kind's two exact tests on what it places: the
// one arithmetic the per-shape functions below and StillShapes share.

PlacedSphere placed_sphere(const contracts::Transform& transform) {
    const Placement p = placement(transform);
    return PlacedSphere{p.t, p.scale};
}

PlacedBox placed_box(const BoxData& box, const contracts::Transform& transform) {
    const Placement p = placement(transform);
    PlacedBox placed;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const int a = static_cast<int>(axis);
        placed.low[axis] = p.t[axis] + p.scale * contracts::component(box.min, a);
        placed.high[axis] = p.t[axis] + p.scale * contracts::component(box.max, a);
    }
    return placed;
}

bool meets(const PlacedSphere& s, const Bounds& box) noexcept {
    double distance2 = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const int a = static_cast<int>(axis);
        const double nearest = std::clamp(s.center[axis], static_cast<double>(contracts::component(box.min, a)),
                                          static_cast<double>(contracts::component(box.max, a)));
        const double d = s.center[axis] - nearest;
        distance2 += d * d;
    }
    return distance2 <= s.radius * s.radius;
}

bool meets(const PlacedBox& b, const Bounds& other) noexcept {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const int a = static_cast<int>(axis);
        if (b.high[axis] < contracts::component(other.min, a) || b.low[axis] > contracts::component(other.max, a)) {
            return false;
        }
    }
    return true;
}

double distance_to(const PlacedSphere& s, contracts::Float3 point) noexcept {
    const std::array<double, 3> d = {point.x - s.center[0], point.y - s.center[1], point.z - s.center[2]};
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - s.radius;
}

double distance_to(const PlacedBox& b, contracts::Float3 point) noexcept {
    double outside2 = 0.0;
    double inside = -std::numeric_limits<double>::infinity();  // the largest (least negative) face distance
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double q = contracts::component(point, static_cast<int>(axis));
        const double below = b.low[axis] - q;
        const double above = q - b.high[axis];
        const double out = std::max({below, above, 0.0});
        outside2 += out * out;
        inside = std::max(inside, std::max(below, above));
    }
    return outside2 > 0.0 ? std::sqrt(outside2) : inside;
}

}  // namespace

bool sphere_touches(const contracts::Transform& transform, const Bounds& box) {
    return meets(placed_sphere(transform), box);
}

bool box_touches(const BoxData& box, const contracts::Transform& transform, const Bounds& other) {
    return meets(placed_box(box, transform), other);
}

double sphere_distance(const contracts::Transform& transform, contracts::Float3 point) {
    return distance_to(placed_sphere(transform), point);
}

double box_distance(const BoxData& box, const contracts::Transform& transform, contracts::Float3 point) {
    return distance_to(placed_box(box, transform), point);
}

Bounds world_bounds(const Bounds& object, const contracts::Transform& transform) {
    const auto& m = transform.m;
    std::array<double, 3> center{};
    std::array<double, 3> half{};
    for (std::size_t j = 0; j < 3; ++j) {
        const int a = static_cast<int>(j);
        const auto lo = static_cast<double>(contracts::component(object.min, a));
        const auto hi = static_cast<double>(contracts::component(object.max, a));
        center[j] = 0.5 * (lo + hi);
        half[j] = 0.5 * (hi - lo);
    }
    std::array<float, 3> lows{};
    std::array<float, 3> highs{};
    for (std::size_t r = 0; r < 3; ++r) {
        double c = m[r][3];
        double e = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            c += static_cast<double>(m[r][j]) * center[j];
            e += std::abs(static_cast<double>(m[r][j])) * half[j];
        }
        // Rounded outward: the float box holds the double one.
        lows[r] = round_down(c - e);
        highs[r] = round_up(c + e);
    }
    return Bounds{{lows[0], lows[1], lows[2]}, {highs[0], highs[1], highs[2]}};
}

// No default in any switch: a kind without bounds or a test fails to
// compile (-Wswitch, -Werror), and a value no kind names throws. .at()
// checks the indices, which the scene reader guarantees; a broken guarantee
// throws rather than reads past.

Bounds object_bounds(const Shapes& shapes, ShapeRecord record) {
    switch (record.kind) {
    case ShapeKind::sphere:
        return sphere_bounds();
    case ShapeKind::box:
        return bounds(shapes.boxes.at(record.geometry));
    }
    no_case("object_bounds");
}

double distance(const Shapes& shapes, std::uint32_t shape, contracts::Float3 point) {
    const ShapeRecord& record = shapes.records.at(shape);
    const contracts::Transform& transform = shapes.transforms.at(shape);
    switch (record.kind) {
    case ShapeKind::sphere:
        return sphere_distance(transform, point);
    case ShapeKind::box:
        return box_distance(shapes.boxes.at(record.geometry), transform, point);
    }
    no_case("distance");
}

bool touches(const Shapes& shapes, std::uint32_t shape, const Bounds& box) {
    const ShapeRecord& record = shapes.records.at(shape);
    const contracts::Transform& transform = shapes.transforms.at(shape);
    switch (record.kind) {
    case ShapeKind::sphere:
        return sphere_touches(transform, box);
    case ShapeKind::box:
        return box_touches(shapes.boxes.at(record.geometry), transform, box);
    }
    no_case("touches");
}

StillShapes::StillShapes(const Shapes& shapes, std::span<const std::uint32_t> which) {
    for (std::uint32_t shape : which) {
        const ShapeRecord& record = shapes.records.at(shape);
        const contracts::Transform& transform = shapes.transforms.at(shape);
        switch (record.kind) {
        case ShapeKind::sphere:
            spheres_.push_back(placed_sphere(transform));
            continue;
        case ShapeKind::box:
            boxes_.push_back(placed_box(shapes.boxes.at(record.geometry), transform));
            continue;
        }
        no_case("StillShapes");
    }
}

double StillShapes::distance(contracts::Float3 point) const noexcept {
    double nearest = std::numeric_limits<double>::infinity();
    for (const PlacedSphere& s : spheres_) {
        nearest = std::min(nearest, distance_to(s, point));
    }
    for (const PlacedBox& b : boxes_) {
        nearest = std::min(nearest, distance_to(b, point));
    }
    return nearest;
}

bool StillShapes::touches(const Bounds& box) const noexcept {
    return std::ranges::any_of(spheres_, [&](const PlacedSphere& s) { return meets(s, box); }) ||
           std::ranges::any_of(boxes_, [&](const PlacedBox& b) { return meets(b, box); });
}

}  // namespace serenity::shapes
