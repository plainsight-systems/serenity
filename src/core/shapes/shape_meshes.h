#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "core/contracts/float3.h"
#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"

namespace serenity::shapes {

// Axis: Acceleration (the shapes as ray-tracing hardware takes them:
// docs/research/2026-10-20-nvidia-hardware-design.md, "The spheres as
// triangles"; docs/research/2026-10-23-apple-silicon-design.md, section 6).
//
// The RTX 3090 Ti's RT cores and the M3's ray-tracing unit intersect
// triangles in hardware, and nothing else, so every shape is a triangle
// mesh in its own space, each shape an instance of its kind's (each
// backend's acceleration/scene_acceleration.h):
//
//   - the unit sphere as an icosphere: the icosahedron, each level
//     splitting every triangle in four and pushing the new vertices out
//     onto the sphere, so every vertex lies on it and every face inside
//     it. Its gap, the deepest a face lies below the sphere, as a fraction
//     of the radius, falls by four a level (its triangles are not all
//     alike, so it is about twice an even edge's sagitta); a sphere's level is
//     chosen by that gap (mesh_levels.h);
//   - the unit box, [0, 1]^3, as 12 triangles: two a face, on the face's
//     plane exactly.
//
// Triangles wind counter-clockwise seen from outside, so a face's normal
// (b - a) x (c - a) points out; nothing is culled by it (rays inside glass
// meet the faces from within), it only fixes one convention.
//
// Plain data made on the host, once, when the structure is made (MEM.9),
// and handed to the backend's builder as they are: a vertex is three
// packed floats (OptiX's OPTIX_VERTEX_FORMAT_FLOAT3, Metal's
// MTLAttributeFormatFloat3), a triangle three 32-bit indices. Pure
// functions of their level (F.8), unit tested without a GPU.
struct TriangleMesh {
  std::vector<contracts::Float3> vertices;
  std::vector<std::array<std::uint32_t, 3>> triangles;
};

// The finest icosphere made: 20 x 4^9 = 5,242,880 triangles, made in a
// fifth of a second, its structure built once. A sphere no level up to it
// serves is refused (mesh_levels.h).
inline constexpr unsigned finest_sphere_level = 9;

// Each level's gap, as a fraction of the radius, levels 0 to
// finest_sphere_level: computed from the meshes icosphere() makes
// (sphere_gap()), written here so a level is chosen without making every
// mesh (Con.5), and checked against them by the unit test.
inline constexpr std::array<double, finest_sphere_level + 1> icosphere_gaps{
    2.053456e-1, 6.582764e-2, 1.775308e-2, 4.528358e-3, 1.137876e-3,
    2.848231e-4, 7.125236e-5, 1.782045e-5, 4.474639e-6, 1.139033e-6};

// The icosphere of `level`, 0 the icosahedron: 10 x 4^level + 2 vertices,
// 20 x 4^level triangles. Throws std::invalid_argument above
// finest_sphere_level (I.6).
TriangleMesh icosphere(unsigned level);

// The deepest any face of `mesh` lies below the unit sphere: 1 less the
// least distance from the center to a face. `mesh`'s vertices must lie on
// the unit sphere (icosphere()'s do).
double sphere_gap(const TriangleMesh& mesh);

// The unit box's 12 triangles: 8 vertices, the corners.
TriangleMesh unit_box();

// a x b, each a 3 x 4 affine matrix (contract 10) with the row
// (0, 0, 0, 1) understood: b first, then a.
contracts::Transform compose(const contracts::Transform& a,
                             const contracts::Transform& b);

// The transform an instance of shape `record`'s mesh takes, the shape
// placed by `placed` (contract 10): its mesh's own placement first, the
// identity for a sphere (the unit sphere's mesh) and the unit box onto the
// box's corners for a box (its geometry `boxes[record.geometry]`), then
// `placed`. The same on every backend (each backend's
// acceleration/scene_acceleration.h). The record must index its kind's
// array, as the scene reader makes it.
contracts::Transform instance_transform(const ShapeRecord& record,
                                        std::span<const BoxData> boxes,
                                        const contracts::Transform& placed);

}  // namespace serenity::shapes
