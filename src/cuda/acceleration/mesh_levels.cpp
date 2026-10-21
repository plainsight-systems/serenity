#include "cuda/acceleration/mesh_levels.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

#include "core/animation/motion.h"
#include "core/contracts/transform.h"
#include "core/shapes/primitive.h"
#include "cuda/acceleration/shape_meshes.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

double distance_to_box(contracts::Float3 p, const contracts::Box& box) {
  const auto gap = [](double v, double lo, double hi) {
    return std::max({lo - v, 0.0, v - hi});
  };
  const double dx = gap(p.x, box.min.x, box.max.x);
  const double dy = gap(p.y, box.min.y, box.max.y);
  const double dz = gap(p.z, box.min.z, box.max.z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// From `p` to the box's farthest corner.
double farthest_in_box(contracts::Float3 p, const contracts::Box& box) {
  const auto far = [](double v, double lo, double hi) {
    return std::max(std::abs(v - lo), std::abs(v - hi));
  };
  const double dx = far(p.x, box.min.x, box.max.x);
  const double dy = far(p.y, box.min.y, box.max.y);
  const double dz = far(p.z, box.min.z, box.max.z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

double nearest_distance(const SphereSight& sphere,
                        const contracts::Camera& camera) {
  // Outside it wherever its center is, the nearest its surface comes is
  // the nearest center less the radius; inside it wherever its center is,
  // the radius less the farthest center; else its surface can pass
  // through the camera.
  const double nearest = distance_to_box(camera.position, sphere.reach);
  const double farthest = farthest_in_box(camera.position, sphere.reach);
  double surface = 0.0;
  if (nearest >= sphere.radius) {
    surface = nearest - sphere.radius;
  } else if (farthest <= sphere.radius) {
    surface = sphere.radius - farthest;
  }
  return std::max(0.0, surface - camera.lens_radius);
}

std::optional<unsigned> sphere_level(const SphereSight& sphere,
                                     const contracts::Camera& camera) {
  const double nearest = nearest_distance(sphere, camera);
  if (!(nearest > 0.0)) {
    return std::nullopt;
  }
  const double half_angle =
      camera.vertical_fov_degrees * std::numbers::pi / 360.0;
  // A pixel's width a unit of distance away, at the frame's middle.
  const double pixel = 2.0 * std::tan(half_angle) / largest_rows;
  for (unsigned level = 0; level <= finest_sphere_level; ++level) {
    const double gap = icosphere_gaps.at(level) * sphere.radius;
    if (gap / nearest <= screen_bound * pixel) {
      return level;
    }
  }
  return std::nullopt;
}

std::vector<std::uint8_t> sphere_levels(const scene::SceneDescription& scene) {
  const std::vector<shapes::ShapeRecord>& records = scene.shapes.records;
  std::vector<std::uint8_t> levels(records.size(), no_level);
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (records[i].kind != shapes::ShapeKind::sphere) {
      continue;
    }
    const contracts::Transform& at_rest = scene.shapes.transforms.at(i);
    const contracts::Float3 center = contracts::translation(at_rest);
    SphereSight sight{.radius = contracts::scale(at_rest),
                      .reach = {center, center}};
    for (const animation::Mover& mover : scene.animation.movers) {
      if (mover.target == i) {
        sight.reach = animation::extent(scene.animation.motions, mover.motion);
      }
    }
    const std::optional<unsigned> level = sphere_level(sight, scene.camera);
    if (!level) {
      throw CudaError(
          "sphere_levels: shape " + std::to_string(i) +
          ", a sphere of radius " + std::to_string(sight.radius) +
          " m: no icosphere up to level " +
          std::to_string(finest_sphere_level) +
          " is within a thirtieth of a pixel from the camera's nearest, " +
          std::to_string(nearest_distance(sight, scene.camera)) + " m");
    }
    levels[i] = static_cast<std::uint8_t>(*level);
  }
  return levels;
}

}  // namespace serenity::cuda
