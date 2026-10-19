#pragma once

// The lit floor the path tracer and ReSTIR PT are held to
// (tests/gpu/path_test.cpp, restir_pt_test.cpp), once
// (ES.3, F.10): a Lambert floor of albedo 0.8 in a black sky under
// fireflies of radius 0.05 and radiance 200, seen from above; and its
// closed form, albedo x L sin^2 cos summed over the lights, at the floor
// point a point of the image sees.

#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/contracts/camera.h"
#include "core/contracts/transform.h"
#include "core/frame/extent.h"
#include "core/scene/scene.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/vector.h"

namespace serenity::tests {

// The floor's albedo, as lit_floor() writes it.
inline constexpr double floor_albedo = 0.8;

// The floor, and `rest` after it: fireflies, or other shapes and their
// materials.
inline std::string lit_floor(std::string_view rest) {
  return camera_text("[0, 2, 3]", "[0, 0, 0]", 50) +
         sky("[0, 0, 0]", "[0, 0, 0]") +
         "[materials.floor]\nkind = \"rough\"\ncolor = [" +
         exact(floor_albedo) + ", " + exact(floor_albedo) + ", " +
         exact(floor_albedo) +
         "]\n"
         "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
         "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, "
         "50]\nmaterial = \"floor\"\n" +
         std::string(rest);
}

// A firefly of the floor's: radius 0.05, its glow 200, at `center`.
inline std::string firefly(std::string_view center) {
  return "[[shapes]]\nkind = \"sphere\"\ncenter = " + std::string(center) +
         "\nradius = 0.05\nmaterial = \"glow\"\n";
}

inline const std::string one_firefly = firefly("[0.3, 2.5, -0.5]");
inline const std::string two_fireflies =
    one_firefly + firefly("[-1.2, 1.5, 0.4]");

// The floor point the image's point (x, y), in pixels from the top left, of
// an image of `size` sees: pixel (i, j)'s center is (i + 0.5, j + 0.5).
inline Vec3 floor_point(const scene::SceneDescription& description,
                        frame::Extent size, double x, double y) {
  const contracts::CameraData c = camera::shader_form(description.camera, size);
  const Vec3 d = camera_direction(c, x, y, size);
  return vec(c.origin) + (-c.origin.y / d.y) * d;
}

// albedo x L sin^2 cos for each of `lights` (their indices among the
// scene's sphere lights) at floor point `p`: what unshadowed fireflies give
// the floor.
inline double floor_from(const scene::SceneDescription& description, Vec3 p,
                         const std::vector<std::size_t>& lights) {
  double sum = 0.0;
  for (const std::size_t i : lights) {
    const lights::SphereLightData& light = description.sphere_lights.at(i);
    // Where its shape is, and how big: the transform's translation and
    // scale (contracts/transform.h).
    const contracts::Transform& placed =
        description.shapes.transforms.at(light.shape);
    const double radius = placed.m[0][0];
    const Vec3 to = vec(contracts::translation(placed)) - p;
    const double d2 = dot(to, to);
    sum += floor_albedo * light.radiance.x * (radius * radius / d2) *
           (to.y / std::sqrt(d2));
  }
  return sum;
}

// Every one of the scene's sphere lights, by index.
inline std::vector<std::size_t> every_light(
    const scene::SceneDescription& description) {
  std::vector<std::size_t> all(description.sphere_lights.size());
  std::iota(all.begin(), all.end(), std::size_t{0});
  return all;
}

// The closed form at the floor point the image's point (x, y) sees, every
// light unshadowed.
inline double expected_floor(const scene::SceneDescription& description,
                             frame::Extent size, double x, double y) {
  return floor_from(description, floor_point(description, size, x, y),
                    every_light(description));
}

}  // namespace serenity::tests
