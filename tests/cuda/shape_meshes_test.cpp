// The shapes' meshes (cuda/acceleration/shape_meshes.h), a unit with no
// driver: each icosphere's counts, its vertices on the unit sphere, its
// surface closed and wound outward, its gap the table's; the unit box's
// twelve triangles on its faces, closed and wound outward; a level past the
// finest refused.

#include "cuda/acceleration/shape_meshes.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

namespace serenity::cuda {
namespace {

struct Edge {
  std::uint32_t from;
  std::uint32_t to;
  auto operator<=>(const Edge&) const = default;
};

// Every directed edge used once and its reverse once: a closed surface,
// its triangles wound alike.
bool closed_and_consistent(const TriangleMesh& mesh) {
  std::map<Edge, int> uses;
  for (const auto& t : mesh.triangles) {
    for (std::size_t i = 0; i < 3; ++i) {
      ++uses[Edge{t.at(i), t.at((i + 1) % 3)}];
    }
  }
  for (const auto& [edge, count] : uses) {
    const auto reverse = uses.find(Edge{edge.to, edge.from});
    if (count != 1 || reverse == uses.end() || reverse->second != 1) {
      return false;
    }
  }
  return true;
}

// Whether every face's normal, (b - a) x (c - a), points away from
// `center`.
bool wound_outward(const TriangleMesh& mesh, contracts::Float3 center) {
  for (const auto& t : mesh.triangles) {
    const contracts::Float3 a = mesh.vertices.at(t[0]);
    const contracts::Float3 b = mesh.vertices.at(t[1]);
    const contracts::Float3 c = mesh.vertices.at(t[2]);
    const double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    const double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double mx = (a.x + b.x + c.x) / 3.0 - center.x;
    const double my = (a.y + b.y + c.y) / 3.0 - center.y;
    const double mz = (a.z + b.z + c.z) / 3.0 - center.z;
    if (!(nx * mx + ny * my + nz * mz > 0.0)) {
      return false;
    }
  }
  return true;
}

TEST_CASE("an icosphere: its counts, on the sphere, closed, wound outward") {
  for (unsigned level = 0; level <= 4; ++level) {
    const TriangleMesh mesh = icosphere(level);
    const std::size_t split = std::size_t{1} << (2u * level);  // 4^level
    CHECK(mesh.vertices.size() == 10 * split + 2);
    CHECK(mesh.triangles.size() == 20 * split);
    for (const contracts::Float3& v : mesh.vertices) {
      CHECK(std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z) ==
            doctest::Approx(1.0).epsilon(1e-6));
    }
    CHECK(closed_and_consistent(mesh));
    CHECK(wound_outward(mesh, {0.0f, 0.0f, 0.0f}));
  }
}

TEST_CASE("each level's gap is the table's, and falls by about four") {
  for (unsigned level = 0; level <= finest_sphere_level; ++level) {
    const double gap = sphere_gap(icosphere(level));
    CHECK(gap == doctest::Approx(icosphere_gaps.at(level)).epsilon(1e-5));
    if (level > 0) {
      const double ratio = icosphere_gaps.at(level - 1) / gap;
      CHECK(ratio > 3.0);
      CHECK(ratio < 4.5);
    }
  }
}

TEST_CASE("a level past the finest is refused") {
  CHECK_THROWS_AS(icosphere(finest_sphere_level + 1), std::invalid_argument);
}

TEST_CASE("the unit box: twelve triangles on its faces, closed, outward") {
  const TriangleMesh box = unit_box();
  CHECK(box.vertices.size() == 8);
  CHECK(box.triangles.size() == 12);
  CHECK(closed_and_consistent(box));
  CHECK(wound_outward(box, {0.5f, 0.5f, 0.5f}));
  // Each triangle on one face's plane: one coordinate the same, 0 or 1,
  // at its three corners.
  for (const auto& t : box.triangles) {
    const contracts::Float3 a = box.vertices.at(t[0]);
    const contracts::Float3 b = box.vertices.at(t[1]);
    const contracts::Float3 c = box.vertices.at(t[2]);
    const bool on_face = (a.x == b.x && b.x == c.x) ||
                         (a.y == b.y && b.y == c.y) ||
                         (a.z == b.z && b.z == c.z);
    CHECK(on_face);
  }
}

}  // namespace
}  // namespace serenity::cuda
