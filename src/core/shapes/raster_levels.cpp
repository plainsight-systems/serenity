#include "core/shapes/raster_levels.h"

#include <array>
#include <cmath>

#include "core/contracts/obstacles.h"
#include "core/shapes/mesh_levels.h"

namespace serenity::shapes {

namespace {

using Vec = std::array<double, 3>;

Vec vec(contracts::Float3 v) { return {v.x, v.y, v.z}; }

double dot(const Vec& a, const Vec& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace

std::optional<unsigned> raster_level(double radius, contracts::Float3 center,
                                     const contracts::Camera& camera) {
  return sphere_level(SphereSight{radius, contracts::Box{center, center}},
                      camera);
}

bool in_view(const contracts::CameraData& camera, contracts::Float3 center,
             double radius) {
  const Vec v{center.x - camera.origin.x, center.y - camera.origin.y,
              center.z - camera.origin.z};
  const Vec forward = vec(camera.forward);
  if (dot(v, forward) < -radius) {
    return false;  // behind the camera
  }
  // A side's plane through the camera, its outward normal a: a point p
  // inside where dot(p - origin, a) <= 0 (the image's x or y within its
  // half extent times the view depth); the sphere where that is at most
  // its radius times |a|.
  const auto inside = [&](const Vec& side, double sign) {
    const double extent2 = dot(side, side);
    Vec a{};
    for (std::size_t c = 0; c < 3; ++c) {
      a.at(c) = sign * side.at(c) / extent2 - forward.at(c);
    }
    return dot(v, a) <= radius * std::sqrt(dot(a, a));
  };
  const Vec right = vec(camera.right);
  const Vec up = vec(camera.up);
  return inside(right, 1.0) && inside(right, -1.0) && inside(up, 1.0) &&
         inside(up, -1.0);
}

}  // namespace serenity::shapes
