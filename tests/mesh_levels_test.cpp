// The spheres' levels (core/shapes/mesh_levels.h), a unit with no
// driver: a sphere's level finer where it is nearer or seen through a
// wider lens, the coarsest that meets the screen's bound; a cavity seen
// from within; none for a sphere whose surface can reach the lens, or one
// too near for the finest level; a
// scene's levels a shape each, a moving sphere's from its motion's reach,
// and a refusal by name.

#include "core/shapes/mesh_levels.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/scene/scene.h"
#include "core/shapes/shape_meshes.h"

namespace serenity::shapes {
namespace {

contracts::Camera camera_at_origin(float lens_radius = 0.0f) {
  contracts::Camera camera;
  camera.position = {0.0f, 0.0f, 0.0f};
  camera.look_at = {0.0f, 0.0f, -1.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.vertical_fov_degrees = 40.0f;
  camera.lens_radius = lens_radius;
  return camera;
}

SphereSight still(contracts::Float3 center, double radius) {
  return SphereSight{radius, {center, center}};
}

// sphere_level()'s answer, which the case requires to be a level.
unsigned required_level(const SphereSight& sphere,
                        const contracts::Camera& camera) {
  const std::optional<unsigned> level = sphere_level(sphere, camera);
  REQUIRE(level.has_value());
  return level.value_or(0u);
}

TEST_CASE("a nearer sphere, or one seen through a wider lens, is finer") {
  const contracts::Camera pinhole = camera_at_origin();
  const unsigned far =
      required_level(still({0.0f, 0.0f, -100.0f}, 1.0), pinhole);
  const unsigned near =
      required_level(still({0.0f, 0.0f, -5.0f}, 1.0), pinhole);
  CHECK(near > far);
  const unsigned through_lens =
      required_level(still({0.0f, 0.0f, -1.2f}, 0.1), camera_at_origin(0.05f));
  const unsigned pinhole_view =
      required_level(still({0.0f, 0.0f, -1.2f}, 0.1), pinhole);
  CHECK(through_lens >= pinhole_view);
  // The level chosen meets the screen's bound, and the one before misses it.
  const double pixel =
      2.0 * std::tan(20.0 * 3.14159265358979 / 180.0) / largest_rows;
  const double nearest =
      nearest_distance(still({0.0f, 0.0f, -5.0f}, 1.0), pinhole);
  CHECK(icosphere_gaps.at(near) / nearest <= screen_bound * pixel);
  CHECK(icosphere_gaps.at(near - 1) / nearest > screen_bound * pixel);
}

TEST_CASE("a camera inside a sphere sees its surface from within") {
  const contracts::Camera camera = camera_at_origin();
  // A cavity 2 m across, the camera 1.5 m from its center: 0.5 m to its
  // wall, as a sphere outside it 0.5 m off.
  const SphereSight cavity = still({0.0f, 0.0f, 1.5f}, 2.0);
  CHECK(nearest_distance(cavity, camera) == doctest::Approx(0.5));
  CHECK(required_level(cavity, camera) ==
        required_level(still({0.0f, 0.0f, -2.5f}, 2.0), camera));
}

TEST_CASE(
    "no level for a sphere whose surface can reach the lens, or one too near") {
  const contracts::Camera camera = camera_at_origin();
  const SphereSight around{0.5, {{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}};
  CHECK_FALSE(sphere_level(around, camera));
  // A meter in radius, its surface 5 cm off: the finest gap is 0.07 of a
  // pixel there.
  CHECK_FALSE(sphere_level(still({0.0f, 0.0f, -1.05f}, 1.0), camera));
}

constexpr const char* header = R"(
[camera]
position = [0, 0, 0]
look_at = [0, 0, -1]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
)";

TEST_CASE("a scene's levels: a sphere's each, a moving one's from its reach") {
  const std::string text = std::string(header) + R"(
[[shapes]]
kind = "box"
min = [-5, -2, -12]
max = [-3, -1, -10]
material = "matte"
[[shapes]]
kind = "sphere"
center = [-2, 0, -10]
radius = 0.5
material = "matte"
[[shapes]]
kind = "sphere"
center = [9, 0, -10]
radius = 0.5
material = "matte"
motion = { kind = "wander", reach = 8.5, speed = 0.01, seed = 1 }
)";
  const scene::SceneDescription description = scene::parse(text, "levels");
  const std::vector<std::uint8_t> levels = sphere_levels(description);
  REQUIRE(levels.size() == 3);
  CHECK(levels[0] == no_level);
  CHECK(levels[1] != no_level);
  CHECK(levels[2] != no_level);
  // Its reach brings it nearer the camera than the still one comes.
  CHECK(levels[2] > levels[1]);
}

TEST_CASE("a sphere no level serves is refused by name") {
  const std::string text = std::string(header) + R"(
[[shapes]]
kind = "sphere"
center = [0, 0, -1.05]
radius = 1
material = "matte"
)";
  const scene::SceneDescription description = scene::parse(text, "refused");
  CHECK(sphere_level_refusal(description)
            .value_or("")
            .find("shape 0, a sphere") != std::string::npos);
  CHECK_THROWS_WITH_AS(sphere_levels(description),
                       doctest::Contains("shape 0, a sphere"),
                       std::invalid_argument);
}

}  // namespace
}  // namespace serenity::shapes
