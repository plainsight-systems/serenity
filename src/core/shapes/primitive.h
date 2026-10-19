#pragma once

// Axis: Shape.
//
// The shape kinds, and the record of each shape in a scene: which geometry
// it is an instance of, and what it is made of. Shared with shaders
// (contracts/frame_constants.h gives the layout rules).
//
// A shape is an instance of a geometry, as a mesh instance is in Unreal: the
// geometry is defined once, in its own coordinates (object space), and each
// shape that is one of it places it in the world with its own transform
// (contract 10, contracts/transform.h). Every sphere in a scene, every marble
// and every firefly, is the one unit sphere (sphere.h), placed at its center
// and scaled by its radius; each box is its own geometry, its corners as the
// file gives them (box.h). A shape that moves changes its transform and nothing
// else. The acceleration structure holds each shape as its geometry's
// bounds placed by its transform (metal/acceleration/scene_acceleration.h).
//
// Shape i of the scene is ShapeRecord i and Transform i, and primitive i of
// the acceleration structure. Its kind says which
// kind's geometry `geometry` indexes; a kind and an index rather than a union
// (Enum.2, C.181). A new kind adds a value here, its geometry's data and
// object-space bounds, and its object-space test; it changes no other kind,
// and nothing outside the Shape family.

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"
#if !defined(__METAL_VERSION__)
#include "core/contracts/obstacles.h"
#endif

namespace serenity {
namespace shapes {

// NOLINTNEXTLINE(performance-enum-size): the layout Metal reads
enum class ShapeKind : uint32_t {
  sphere =
      0,    // the unit sphere (sphere.h): no data of its own; `geometry` is 0
  box = 1,  // an axis-aligned box, min to max (box.h): `geometry` indexes the
            // boxes
};

struct ShapeRecord {
  ShapeKind kind;
  uint32_t geometry;  // into that kind's geometry array
  uint32_t material;  // index into the material records
  // the medium inside it (contract 12): an index into the medium records
  // (media/medium.h), or contracts::no_medium for air
  uint32_t interior;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(ShapeRecord) == 16,
    "ShapeRecord must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ShapeRecord>,
              "ShapeRecord is written to the GPU as bytes");
#endif

#if !defined(__METAL_VERSION__)
// An axis-aligned box: a geometry's extent in object space, what its
// acceleration structure holds; or a shape's extent in the world (shapes.h).
// Two packed triples, the layout of Metal's and Vulkan's bounding boxes
// alike. Contract 11's box (contracts/obstacles.h), the same two corners,
// not a second type of them (ES.3), as animation/extent.h takes it too: the
// still shapes answer contract 11's touches() with it as it is.
using Bounds = contracts::Box;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
static_assert(sizeof(Bounds) == 24,
              "Bounds must be two packed triples of floats");
static_assert(std::is_trivially_copyable_v<Bounds>,
              "Bounds is written to the GPU as bytes");
#endif

}  // namespace shapes
}  // namespace serenity
