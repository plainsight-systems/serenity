#include "core/shapes/shape_meshes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace serenity::shapes {

namespace {

struct Vec {
  double x, y, z;
};

Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec operator*(double s, Vec a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec unit(Vec a) { return (1.0 / std::sqrt(dot(a, a))) * a; }

Vec of(contracts::Float3 p) { return {p.x, p.y, p.z}; }

contracts::Float3 packed(Vec a) {
  return {static_cast<float>(a.x), static_cast<float>(a.y),
          static_cast<float>(a.z)};
}

// The point of triangle (a, b, c) nearest the origin (Ericson, Real-Time
// Collision Detection, 5.1.5: the closest point on a triangle, by its
// Voronoi regions), in double.
Vec nearest_to_origin(Vec a, Vec b, Vec c) {
  const Vec p{0.0, 0.0, 0.0};
  const Vec ab = b - a;
  const Vec ac = c - a;
  const Vec ap = p - a;
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return a;
  }
  const Vec bp = p - b;
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return b;
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    return a + (d1 / (d1 - d3)) * ab;
  }
  const Vec cp = p - c;
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return c;
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    return a + (d2 / (d2 - d6)) * ac;
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
  }
  const double denominator = 1.0 / (va + vb + vc);
  return a + (vb * denominator) * ab + (vc * denominator) * ac;
}

// The icosahedron's 12 vertices on the unit sphere and its 20 faces,
// counter-clockwise from outside (shape_meshes.h).
TriangleMesh icosahedron() {
  // The golden ratio's corners and faces: a table, its numbers its own.
  // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  const std::array<Vec, 12> corners{{{-1, phi, 0},
                                     {1, phi, 0},
                                     {-1, -phi, 0},
                                     {1, -phi, 0},
                                     {0, -1, phi},
                                     {0, 1, phi},
                                     {0, -1, -phi},
                                     {0, 1, -phi},
                                     {phi, 0, -1},
                                     {phi, 0, 1},
                                     {-phi, 0, -1},
                                     {-phi, 0, 1}}};
  TriangleMesh mesh;
  mesh.vertices.reserve(corners.size());
  for (const Vec& corner : corners) {
    mesh.vertices.push_back(packed(unit(corner)));
  }
  mesh.triangles = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10},
                    {0, 10, 11}, {1, 5, 9},  {5, 11, 4}, {11, 10, 2},
                    {10, 7, 6},  {7, 1, 8},  {3, 9, 4},  {3, 4, 2},
                    {3, 2, 6},   {3, 6, 8},  {3, 8, 9},  {4, 9, 5},
                    {2, 4, 11},  {6, 2, 10}, {8, 6, 7},  {9, 8, 1}};
  // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
  return mesh;
}

// `mesh` with every triangle split in four, each edge's midpoint pushed
// out onto the unit sphere and shared by the edge's two triangles.
TriangleMesh split(const TriangleMesh& mesh) {
  TriangleMesh finer;
  finer.vertices = mesh.vertices;
  finer.vertices.reserve(mesh.vertices.size() + mesh.triangles.size() * 3 / 2);
  finer.triangles.reserve(mesh.triangles.size() * 4);
  std::unordered_map<std::uint64_t, std::uint32_t> midpoints;
  midpoints.reserve(mesh.triangles.size() * 3 / 2);
  const auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
    const std::uint64_t key =
        (std::uint64_t{std::min(a, b)} << 32u) | std::max(a, b);
    const auto [at, made] = midpoints.try_emplace(
        key, static_cast<std::uint32_t>(finer.vertices.size()));
    if (made) {
      // Along the sum's direction, the midpoint's (halving it changes
      // nothing a normalization keeps), pushed out onto the sphere.
      finer.vertices.push_back(
          packed(unit(of(mesh.vertices[a]) + of(mesh.vertices[b]))));
    }
    return at->second;
  };
  for (const std::array<std::uint32_t, 3>& t : mesh.triangles) {
    const std::uint32_t ab = midpoint(t[0], t[1]);
    const std::uint32_t bc = midpoint(t[1], t[2]);
    const std::uint32_t ca = midpoint(t[2], t[0]);
    finer.triangles.push_back({t[0], ab, ca});
    finer.triangles.push_back({t[1], bc, ab});
    finer.triangles.push_back({t[2], ca, bc});
    finer.triangles.push_back({ab, bc, ca});
  }
  return finer;
}

}  // namespace

TriangleMesh icosphere(unsigned level) {
  if (level > finest_sphere_level) {
    throw std::invalid_argument("icosphere: level " + std::to_string(level) +
                                " is past the finest, " +
                                std::to_string(finest_sphere_level));
  }
  TriangleMesh mesh = icosahedron();
  for (unsigned i = 0; i < level; ++i) {
    mesh = split(mesh);
  }
  return mesh;
}

double sphere_gap(const TriangleMesh& mesh) {
  double least = 1.0;
  for (const std::array<std::uint32_t, 3>& t : mesh.triangles) {
    const Vec nearest =
        nearest_to_origin(of(mesh.vertices[t[0]]), of(mesh.vertices[t[1]]),
                          of(mesh.vertices[t[2]]));
    least = std::min(least, std::sqrt(dot(nearest, nearest)));
  }
  return 1.0 - least;
}

TriangleMesh unit_box() {
  TriangleMesh mesh;
  // Corner i has x = bit 0, y = bit 1, z = bit 2.
  constexpr std::uint32_t corners = 8;
  for (std::uint32_t i = 0; i < corners; ++i) {
    mesh.vertices.push_back({static_cast<float>(i & 1u),
                             static_cast<float>((i >> 1u) & 1u),
                             static_cast<float>((i >> 2u) & 1u)});
  }
  // Two a face, counter-clockwise from outside: -x, +x, -y, +y, -z, +z;
  // a table of corners.
  // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
  mesh.triangles = {{0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5},
                    {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
                    {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}};
  // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
  return mesh;
}

contracts::Transform compose(const contracts::Transform& a,
                             const contracts::Transform& b) {
  contracts::Transform c{};
  // Contract 10's matrix is a C array (MSL has no std::array), indexed by
  // these loops' bounds.
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 4; ++k) {
      float sum = k == 3 ? a.m[r][3] : 0.0f;
      for (int j = 0; j < 3; ++j) {
        sum += a.m[r][j] * b.m[j][k];
      }
      c.m[r][k] = sum;
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  return c;
}

contracts::Transform instance_transform(const ShapeRecord& record,
                                        std::span<const BoxData> boxes,
                                        const contracts::Transform& placed) {
  contracts::Transform mesh{{
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  }};  // a sphere's: the unit sphere's
  if (record.kind == ShapeKind::box) {
    // The unit box onto the box's corners, in its own space.
    const BoxData& box = boxes[record.geometry];
    mesh = contracts::Transform{{
        {box.max.x - box.min.x, 0.0f, 0.0f, box.min.x},
        {0.0f, box.max.y - box.min.y, 0.0f, box.min.y},
        {0.0f, 0.0f, box.max.z - box.min.z, box.min.z},
    }};
  }
  return compose(placed, mesh);
}

}  // namespace serenity::shapes
