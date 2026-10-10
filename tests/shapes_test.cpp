// Every shape as an instance of a geometry: its object-space bounds, and
// whether it touches a world box, whatever its kind.

#include <cmath>
#include <stdexcept>

#include <doctest/doctest.h>

#include "core/shapes/shapes.h"

using namespace serenity;
using namespace serenity::shapes;

namespace {

Shapes two_spheres_and_a_box() {
    Shapes shapes;
    shapes.boxes.push_back(BoxData{{-1.0f, -2.0f, -3.0f}, 0, {1.0f, 2.0f, 3.0f}, 0});
    shapes.records = {{ShapeKind::sphere, 0, 0, 0}, {ShapeKind::box, 0, 0, 0}, {ShapeKind::sphere, 0, 0, 0}};
    shapes.transforms = {contracts::placed({1.0f, 2.0f, 3.0f}, 0.5f), contracts::placed({10.0f, 0.0f, 0.0f}, 1.0f),
                         contracts::placed({-4.0f, 0.0f, 0.0f}, 2.0f)};
    return shapes;
}

}  // namespace

TEST_CASE("a geometry's bounds are in its own space: the unit sphere's, the box's corners") {
    const Shapes shapes = two_spheres_and_a_box();
    const Bounds sphere = object_bounds(shapes, shapes.records[0]);
    CHECK(sphere.min.x == -1.0f);
    CHECK(sphere.max.z == 1.0f);
    const Bounds box = object_bounds(shapes, shapes.records[1]);
    CHECK(box.min.x == -1.0f);
    CHECK(box.max.y == 2.0f);
    CHECK(box.min.z == -3.0f);
}

TEST_CASE("a sphere touches a box exactly when the nearest point of the box is within its radius") {
    const Shapes shapes = two_spheres_and_a_box();
    // Sphere 0: center (1, 2, 3), radius 0.5. A box whose face is 0.5 away
    // along x touches; a hair further does not.
    CHECK(touches(shapes, 0, Bounds{{1.5f, 0.0f, 0.0f}, {2.0f, 4.0f, 6.0f}}));
    CHECK_FALSE(touches(shapes, 0, Bounds{{1.5001f, 0.0f, 0.0f}, {2.0f, 4.0f, 6.0f}}));
    // Toward a corner: the nearest point is (1.4, 2.4, 3), at 0.4 sqrt(2) =
    // 0.566 from the center, beyond the radius, though the box overlaps the
    // sphere's bounding box.
    CHECK_FALSE(touches(shapes, 0, Bounds{{1.4f, 2.4f, 0.0f}, {2.0f, 3.0f, 6.0f}}));
    // A box around the center touches.
    CHECK(touches(shapes, 0, Bounds{{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}}));
}

TEST_CASE("a box touches a box when they overlap or meet on every axis") {
    const Shapes shapes = two_spheres_and_a_box();
    // Shape 1: the box from (-1, -2, -3) to (1, 2, 3) placed at (10, 0, 0):
    // x in [9, 11].
    CHECK(touches(shapes, 1, Bounds{{11.0f, 0.0f, 0.0f}, {12.0f, 1.0f, 1.0f}}));
    CHECK_FALSE(touches(shapes, 1, Bounds{{11.001f, 0.0f, 0.0f}, {12.0f, 1.0f, 1.0f}}));
    CHECK_FALSE(touches(shapes, 1, Bounds{{9.0f, 2.5f, 0.0f}, {10.0f, 3.0f, 1.0f}}));
}

TEST_CASE("a rotated transform, or a record past its kind's array, is refused, not read") {
    Shapes shapes = two_spheres_and_a_box();
    shapes.transforms[0].m[0][1] = 0.5f;
    CHECK_THROWS(touches(shapes, 0, Bounds{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}));

    Shapes broken;
    broken.records = {{ShapeKind::box, 3, 0, 0}};
    broken.transforms = {contracts::placed({0.0f, 0.0f, 0.0f}, 1.0f)};
    CHECK_THROWS(object_bounds(broken, broken.records[0]));
    CHECK_THROWS(touches(broken, 1, Bounds{}));
}

TEST_CASE("a placed geometry's world box: the tight box of the transformed box, rounded outward") {
    // The unit sphere scaled by 0.03 and moved to (1, 2, 3).
    const Bounds sphere = world_bounds(sphere_bounds(), contracts::placed({1.0f, 2.0f, 3.0f}, 0.03f));
    CHECK(double(sphere.min.x) <= 1.0 - 0.03);
    CHECK(double(sphere.max.z) >= 3.0 + 0.03);
    CHECK(sphere.max.z - 3.03f < 1e-6f);
    // A box of half extent (2, 1, 1) turned 90 degrees about z: its x and y
    // extents swap.
    contracts::Transform turned{{{0.0f, -1.0f, 0.0f, 5.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
    const Bounds box = world_bounds(Bounds{{-2.0f, -1.0f, -1.0f}, {2.0f, 1.0f, 1.0f}}, turned);
    CHECK(box.min.x == 4.0f);
    CHECK(box.max.x == 6.0f);
    CHECK(box.min.y == -2.0f);
    CHECK(box.max.y == 2.0f);
    // Turned 45 degrees: the tight box of the turned box, half extent
    // (2 + 1) / sqrt(2) on x and y.
    const float c = std::sqrt(0.5f);
    contracts::Transform diagonal{{{c, -c, 0.0f, 0.0f}, {c, c, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
    const Bounds turned_box = world_bounds(Bounds{{-2.0f, -1.0f, -1.0f}, {2.0f, 1.0f, 1.0f}}, diagonal);
    CHECK(turned_box.max.x == doctest::Approx(3.0 * std::sqrt(0.5)).epsilon(1e-6));
    CHECK(double(turned_box.max.x) >= 3.0 * double(c));
}

TEST_CASE("a record whose kind is no ShapeKind is refused, never answered as a shape") {
    Shapes shapes = two_spheres_and_a_box();
    shapes.records[0].kind = static_cast<ShapeKind>(7);
    CHECK_THROWS_AS(distance(shapes, 0, {0.0f, 0.0f, 0.0f}), std::logic_error);
    CHECK_THROWS_AS(touches(shapes, 0, Bounds{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}), std::logic_error);
    CHECK_THROWS_AS(object_bounds(shapes, shapes.records[0]), std::logic_error);
}
