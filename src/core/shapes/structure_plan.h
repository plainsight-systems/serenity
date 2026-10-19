#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"
#include "core/shapes/shapes.h"

namespace serenity::shapes {

// Axis: Acceleration (what each structure holds, for every backend).
//
// The structures rays are traced against, as the shapes they hold, before
// any backend builds them (Metal's: metal/acceleration/scene_acceleration.h,
// which gives the design and its measurements): one structure of boxes a
// ray mask finds (contract 19, core/contracts/ray_mask.h), each box a
// member shape's geometry's object-space bounds placed in the world by its
// transform (contract 10).
//
//   everything   every shape, so box i is shape i;
//   occluders    the shapes that are not lights, in shape order, its own
//                structure only when the scene has lights and surfaces
//                both: with no light it is everything, with no surface it
//                holds nothing. Its table names each box's shape.
//
// For each, which of its boxes move (their shapes are among `moving`),
// those shapes, their object-space bounds, and every box at rest; and each
// frame, the moving boxes placed where the frame's transforms put their
// shapes (place_moving).
//
// The core's (A.1): which shapes a structure holds and where its boxes are
// is the same on every backend; how a backend builds, keeps and rebuilds a
// structure is its own. Pure functions (F.8).

// One structure's shapes: its box i is shape members[i].
struct TreePlan {
  std::vector<std::uint32_t> members;  // increasing
  // the boxes of its moving shapes, which each frame places
  std::vector<std::uint32_t> moving_boxes;
  // their shapes, whose transforms place them
  std::vector<std::uint32_t> moving_shapes;
  // their object-space bounds, in the same order
  std::vector<Bounds> moving_bounds;
  // every member's box at rest: its bounds placed by its transform
  std::vector<Bounds> at_rest;

  bool moves() const noexcept { return !moving_boxes.empty(); }
};

struct StructurePlan {
  TreePlan everything;
  std::optional<TreePlan> occluders;  // its own, when it differs (above)
  // the shapes a ray carrying the occluders finds, by its structure's box
  std::vector<std::uint32_t> occluder_table;
};

// Shapes that no structure can be planned over, or roles that are not
// shapes' indices: no shapes, not one transform a shape, or `moving` or
// `lights` not indices of shapes in increasing order (I.5). The reason, or
// none.
std::optional<std::string> structure_refusal(
    const Shapes& shapes, std::span<const std::uint32_t> moving,
    std::span<const std::uint32_t> lights);

// The structures over `shapes`, of which `moving` move and `lights` are
// lights, each by index in increasing order. Precondition:
// structure_refusal() gives none; throws std::invalid_argument otherwise.
StructurePlan plan_structures(const Shapes& shapes,
                              std::span<const std::uint32_t> moving,
                              std::span<const std::uint32_t> lights);

// Into `boxes`, `tree`'s boxes, each of its moving boxes where `transforms`
// (one a shape) places its shape; the still ones are not written.
// Precondition: `boxes` is one a member and `transforms` one a shape.
void place_moving(const TreePlan& tree,
                  std::span<const contracts::Transform> transforms,
                  std::span<Bounds> boxes);

}  // namespace serenity::shapes
