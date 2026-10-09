// Every shape's bounds, in primitive order, whatever its kind.

#include <doctest/doctest.h>

#include "core/shapes/shapes.h"

using namespace serenity::shapes;

TEST_CASE("bounds follow the records' order, across kinds") {
    Shapes shapes;
    shapes.spheres.push_back(SphereData{{1.0f, 2.0f, 3.0f}, 0.5f, 0, {}});
    shapes.spheres.push_back(SphereData{{-4.0f, 0.0f, 0.0f}, 2.0f, 0, {}});
    shapes.boxes.push_back(BoxData{{-1.0f, -2.0f, -3.0f}, 0, {1.0f, 2.0f, 3.0f}, 0});
    shapes.records = {{ShapeKind::sphere, 1}, {ShapeKind::box, 0}, {ShapeKind::sphere, 0}};

    const std::vector<Bounds> all = bounds(shapes);
    REQUIRE(all.size() == 3);
    CHECK(all[0].min.x == -6.0f);
    CHECK(all[0].max.y == 2.0f);
    CHECK(all[1].min.z == -3.0f);
    CHECK(all[1].max.x == 1.0f);
    CHECK(all[2].min.x == 0.5f);
    CHECK(all[2].max.z == 3.5f);
}

TEST_CASE("a record past its kind's array is refused, not read") {
    Shapes shapes;
    shapes.records = {{ShapeKind::box, 0}};
    CHECK_THROWS(bounds(shapes));
}
