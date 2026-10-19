// The acceleration structures' plan (core/shapes/structure_plan.h): which
// shapes each structure holds, which of its boxes move, the occluders'
// table, and the roles it refuses.

#include "core/shapes/structure_plan.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "core/scene/scene.h"
#include "support/scene_text.h"

namespace serenity::shapes {
namespace {

using Indices = std::vector<std::uint32_t>;

Shapes some_shapes() {
  return scene::parse(tests::flying_fireflies(), "flying fireflies").shapes;
}

Indices first(std::size_t n) {
  Indices every(n);
  for (std::size_t i = 0; i < n; ++i) {
    every[i] = static_cast<std::uint32_t>(i);
  }
  return every;
}

bool same(const Bounds& a, const Bounds& b) {
  return a.min.x == b.min.x && a.min.y == b.min.y && a.min.z == b.min.z &&
         a.max.x == b.max.x && a.max.y == b.max.y && a.max.z == b.max.z;
}

TEST_CASE(
    "roles that are not shapes' indices in increasing order are refused") {
  const Shapes s = some_shapes();
  REQUIRE(s.records.size() >= 4);
  const auto n = static_cast<std::uint32_t>(s.records.size());
  CHECK_FALSE(structure_refusal(s, Indices{1, 3}, Indices{2}));
  CHECK(structure_refusal(s, Indices{3, 1}, Indices{}));
  CHECK(structure_refusal(s, Indices{1, 1}, Indices{}));
  CHECK(structure_refusal(s, Indices{}, Indices{n}));
  CHECK(structure_refusal(Shapes{}, Indices{}, Indices{}));
  Shapes short_of_one = s;
  short_of_one.transforms.pop_back();
  CHECK(structure_refusal(short_of_one, Indices{}, Indices{}));
}

TEST_CASE(
    "everything holds every shape; the occluders, the shapes not lights") {
  const Shapes s = some_shapes();
  const std::size_t n = s.records.size();
  const StructurePlan plan = plan_structures(s, Indices{}, Indices{0, 2});
  CHECK(plan.everything.members == first(n));
  REQUIRE(plan.occluders);
  Indices surfaces = first(n);
  surfaces.erase(surfaces.begin() + 2);
  surfaces.erase(surfaces.begin());
  CHECK(plan.occluders->members == surfaces);
  CHECK(plan.occluder_table == surfaces);
  CHECK(plan.everything.at_rest.size() == n);
  CHECK(plan.occluders->at_rest.size() == surfaces.size());
  for (std::size_t box = 0; box < surfaces.size(); ++box) {
    const std::uint32_t shape = surfaces[box];
    CHECK(same(
        plan.occluders->at_rest[box],
        world_bounds(object_bounds(s, s.records[shape]), s.transforms[shape])));
  }
}

TEST_CASE("with no light the occluders are everything; with no surface, none") {
  const Shapes s = some_shapes();
  const std::size_t n = s.records.size();
  const StructurePlan unlit = plan_structures(s, Indices{}, Indices{});
  CHECK_FALSE(unlit.occluders);
  CHECK(unlit.occluder_table == first(n));
  const StructurePlan all_lights = plan_structures(s, Indices{}, first(n));
  CHECK_FALSE(all_lights.occluders);
  CHECK(all_lights.occluder_table.empty());
}

TEST_CASE(
    "a structure's moving boxes are its moving shapes', and only they move") {
  const Shapes s = some_shapes();
  const StructurePlan plan = plan_structures(s, Indices{1, 3}, Indices{3});
  CHECK(plan.everything.moving_boxes == Indices{1, 3});
  CHECK(plan.everything.moving_shapes == Indices{1, 3});
  REQUIRE(plan.occluders);
  CHECK(plan.occluders->moving_shapes == Indices{1});
  CHECK(plan.occluders->moving_boxes == Indices{1});  // shape 1 is box 1
  CHECK_FALSE(plan_structures(s, Indices{}, Indices{3}).everything.moves());

  // Placed where a frame's transforms put them; the still boxes untouched.
  std::vector<contracts::Transform> moved = s.transforms;
  moved[1].m[0][3] += 2.0f;
  std::vector<Bounds> boxes = plan.everything.at_rest;
  place_moving(plan.everything, moved, boxes);
  CHECK(same(boxes[1], world_bounds(object_bounds(s, s.records[1]), moved[1])));
  CHECK_FALSE(same(boxes[1], plan.everything.at_rest[1]));
  CHECK(same(boxes[0], plan.everything.at_rest[0]));
  CHECK(same(boxes[2], plan.everything.at_rest[2]));
}

}  // namespace
}  // namespace serenity::shapes
