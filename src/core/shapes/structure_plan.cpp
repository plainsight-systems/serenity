#include "core/shapes/structure_plan.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace serenity::shapes {

namespace {

// Whether `indices` are shapes' indices, of `count`, in increasing order.
bool increasing_shapes(std::span<const std::uint32_t> indices,
                       std::size_t count) {
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= count || (i > 0 && indices[i] <= indices[i - 1])) {
      return false;
    }
  }
  return true;
}

// A tree over `members`, the shapes it holds in increasing order, of which
// `moving` (increasing) move.
TreePlan plan_tree(const Shapes& shapes, std::vector<std::uint32_t> members,
                   std::span<const std::uint32_t> moving) {
  TreePlan tree;
  tree.members = std::move(members);
  const std::vector<std::uint32_t>& m = tree.members;
  // Its moving shapes, by the box each is in: both lists increasing, so one
  // walk pairs them.
  std::size_t next = 0;
  for (std::size_t box = 0; box < m.size(); ++box) {
    while (next < moving.size() && moving[next] < m[box]) {
      ++next;
    }
    if (next < moving.size() && moving[next] == m[box]) {
      tree.moving_boxes.push_back(static_cast<std::uint32_t>(box));
      tree.moving_shapes.push_back(m[box]);
      tree.moving_bounds.push_back(
          object_bounds(shapes, shapes.records[m[box]]));
    }
  }
  // Every member's box at rest: its geometry's bounds placed by its
  // transform.
  tree.at_rest.resize(m.size());
  for (std::size_t box = 0; box < m.size(); ++box) {
    const std::uint32_t shape = m[box];
    tree.at_rest[box] = world_bounds(
        object_bounds(shapes, shapes.records[shape]), shapes.transforms[shape]);
  }
  return tree;
}

}  // namespace

std::optional<std::string> structure_refusal(
    const Shapes& shapes, std::span<const std::uint32_t> moving,
    std::span<const std::uint32_t> lights) {
  const std::size_t count = shapes.records.size();
  if (count == 0 || shapes.transforms.size() != count) {
    return "no shapes, or not one transform per shape";
  }
  if (!increasing_shapes(moving, count)) {
    return "the moving shapes are not shapes' indices in increasing order";
  }
  if (!increasing_shapes(lights, count)) {
    return "the lights are not shapes' indices in increasing order";
  }
  return std::nullopt;
}

StructurePlan plan_structures(const Shapes& shapes,
                              std::span<const std::uint32_t> moving,
                              std::span<const std::uint32_t> lights) {
  if (const std::optional<std::string> reason =
          structure_refusal(shapes, moving, lights)) {
    throw std::invalid_argument("plan_structures: " + *reason);
  }
  const std::size_t count = shapes.records.size();
  // Everything's members are every shape, so its box i is shape i; the
  // occluders', the shapes that are not lights, in order.
  std::vector<std::uint32_t> every(count);
  for (std::size_t i = 0; i < count; ++i) {
    every[i] = static_cast<std::uint32_t>(i);
  }
  std::vector<std::uint32_t> surfaces;
  std::ranges::set_difference(every, lights, std::back_inserter(surfaces));

  StructurePlan plan;
  plan.occluder_table = lights.empty() ? every : surfaces;
  if (!lights.empty() && !surfaces.empty()) {
    plan.occluders = plan_tree(shapes, surfaces, moving);
  }
  plan.everything = plan_tree(shapes, std::move(every), moving);
  return plan;
}

void place_moving(const TreePlan& tree,
                  std::span<const contracts::Transform> transforms,
                  std::span<Bounds> boxes) {
  for (std::size_t i = 0; i < tree.moving_boxes.size(); ++i) {
    boxes[tree.moving_boxes[i]] =
        world_bounds(tree.moving_bounds[i], transforms[tree.moving_shapes[i]]);
  }
}

}  // namespace serenity::shapes
