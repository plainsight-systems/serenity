#include "core/shapes/shapes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace serenity::shapes {

namespace {

// A transform's translation and uniform scale, in double, refusing one with
// a rotation: the world-space tests below rely on a placed box staying
// axis-aligned, which the scene format guarantees (contracts/transform.h).
struct Placement {
    double t[3];
    double scale;
};

Placement placement(const contracts::Transform& transform) {
    const auto& m = transform.m;
    if (m[0][1] != 0.0f || m[0][2] != 0.0f || m[1][0] != 0.0f || m[1][2] != 0.0f || m[2][0] != 0.0f ||
        m[2][1] != 0.0f || m[0][0] != m[1][1] || m[1][1] != m[2][2] || !(m[0][0] > 0.0f)) {
        throw std::invalid_argument("a shape's transform has a rotation; the world-space tests take none");
    }
    return Placement{{m[0][3], m[1][3], m[2][3]}, m[0][0]};
}

double lo(const Bounds& b, int axis) {
    return axis == 0 ? b.min.x : (axis == 1 ? b.min.y : b.min.z);
}

double hi(const Bounds& b, int axis) {
    return axis == 0 ? b.max.x : (axis == 1 ? b.max.y : b.max.z);
}

double low(const BoxData& box, int axis) {
    return axis == 0 ? box.min.x : (axis == 1 ? box.min.y : box.min.z);
}

double high(const BoxData& box, int axis) {
    return axis == 0 ? box.max.x : (axis == 1 ? box.max.y : box.max.z);
}

}  // namespace

bool sphere_touches(const contracts::Transform& transform, const Bounds& box) {
    const Placement p = placement(transform);
    double distance2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double nearest = std::clamp(p.t[axis], lo(box, axis), hi(box, axis));
        const double d = p.t[axis] - nearest;
        distance2 += d * d;
    }
    return distance2 <= p.scale * p.scale;
}

bool box_touches(const BoxData& box, const contracts::Transform& transform, const Bounds& other) {
    const Placement p = placement(transform);
    for (int axis = 0; axis < 3; ++axis) {
        const double placed_low = p.t[axis] + p.scale * low(box, axis);
        const double placed_high = p.t[axis] + p.scale * high(box, axis);
        if (placed_high < lo(other, axis) || placed_low > hi(other, axis)) {
            return false;
        }
    }
    return true;
}

double sphere_distance(const contracts::Transform& transform, contracts::Float3 point) {
    const Placement p = placement(transform);
    const double d[3] = {point.x - p.t[0], point.y - p.t[1], point.z - p.t[2]};
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - p.scale;
}

double box_distance(const BoxData& box, const contracts::Transform& transform, contracts::Float3 point) {
    const Placement p = placement(transform);
    const double q[3] = {point.x, point.y, point.z};
    double outside2 = 0.0;
    double inside = -std::numeric_limits<double>::infinity();  // the largest (least negative) face distance
    for (int axis = 0; axis < 3; ++axis) {
        const double lo_face = p.t[axis] + p.scale * low(box, axis);
        const double hi_face = p.t[axis] + p.scale * high(box, axis);
        const double below = lo_face - q[axis];
        const double above = q[axis] - hi_face;
        const double out = std::max({below, above, 0.0});
        outside2 += out * out;
        inside = std::max(inside, std::max(below, above));
    }
    return outside2 > 0.0 ? std::sqrt(outside2) : inside;
}

Bounds world_bounds(const Bounds& object, const contracts::Transform& transform) {
    const auto& m = transform.m;
    const double center[3] = {0.5 * (double(object.min.x) + object.max.x), 0.5 * (double(object.min.y) + object.max.y),
                              0.5 * (double(object.min.z) + object.max.z)};
    const double half[3] = {0.5 * (double(object.max.x) - object.min.x), 0.5 * (double(object.max.y) - object.min.y),
                            0.5 * (double(object.max.z) - object.min.z)};
    float lows[3];
    float highs[3];
    for (int r = 0; r < 3; ++r) {
        double c = m[r][3];
        double e = 0.0;
        for (int j = 0; j < 3; ++j) {
            c += double(m[r][j]) * center[j];
            e += std::abs(double(m[r][j])) * half[j];
        }
        // Rounded outward: the float box holds the double one.
        const double low = c - e;
        const double high = c + e;
        lows[r] = static_cast<float>(low);
        if (double(lows[r]) > low) {
            lows[r] = std::nextafter(lows[r], -std::numeric_limits<float>::infinity());
        }
        highs[r] = static_cast<float>(high);
        if (double(highs[r]) < high) {
            highs[r] = std::nextafter(highs[r], std::numeric_limits<float>::infinity());
        }
    }
    return Bounds{{lows[0], lows[1], lows[2]}, {highs[0], highs[1], highs[2]}};
}

// No default in either switch: a kind without bounds or a test fails to
// compile (-Wswitch, -Werror). .at() checks the indices, which the scene
// reader guarantees; a broken guarantee throws rather than reads past.

Bounds object_bounds(const Shapes& shapes, const ShapeRecord& record) {
    switch (record.kind) {
    case ShapeKind::sphere:
        return sphere_bounds();
    case ShapeKind::box:
        return bounds(shapes.boxes.at(record.geometry));
    }
    return {};
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
    return 0.0;
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
    return false;
}

}  // namespace serenity::shapes
