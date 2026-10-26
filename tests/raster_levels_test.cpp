// The rasterizer's levels and view (core/shapes/raster_levels.h), a unit
// with no driver: a sphere's level this frame the tracer's rule seen from
// where it is, coarser farther off and never finer than the level its
// nearest approach takes; in view or not by each of the frame's four sides
// and behind the camera, a sphere straddling a side kept.

#include "core/shapes/raster_levels.h"

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/contracts/obstacles.h"
#include "core/shapes/mesh_levels.h"
#include "support/required.h"

namespace serenity::shapes {
namespace {

contracts::Camera looking_down_z() {
  contracts::Camera camera;
  camera.position = {0.0f, 0.0f, 0.0f};
  camera.look_at = {0.0f, 0.0f, -1.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.vertical_fov_degrees = 40.0f;
  return camera;
}

}  // namespace

TEST_CASE(
    "a sphere's raster level is the tracer's rule seen from where it is") {
  const contracts::Camera camera = looking_down_z();
  const auto near = raster_level(0.01, {0.0f, 0.0f, -0.5f}, camera);
  const auto far = raster_level(0.01, {0.0f, 0.0f, -5.0f}, camera);
  REQUIRE(near);
  REQUIRE(far);
  CHECK(tests::required(far) < tests::required(near));
  // As the tracer's for a sphere that stays put.
  const auto staying = sphere_level(
      SphereSight{0.01,
                  contracts::Box{{0.0f, 0.0f, -0.5f}, {0.0f, 0.0f, -0.5f}}},
      camera);
  REQUIRE(staying);
  CHECK(tests::required(near) == tests::required(staying));
  // Never finer than the level of its nearest approach, the tracer's.
  const auto nearest = sphere_level(
      SphereSight{0.01,
                  contracts::Box{{0.0f, 0.0f, -5.0f}, {0.0f, 0.0f, -0.5f}}},
      camera);
  REQUIRE(nearest);
  CHECK(tests::required(far) <= tests::required(nearest));
  // A sphere whose surface touches the lens has none.
  CHECK_FALSE(raster_level(0.01, {0.0f, 0.0f, -0.01f}, camera));
}

TEST_CASE("a sphere is in view inside the frame's four sides and ahead") {
  const contracts::CameraData camera =
      camera::shader_form(looking_down_z(), {400, 300});
  CHECK(in_view(camera, {0.0f, 0.0f, -2.0f}, 0.01));
  // Behind the camera.
  CHECK_FALSE(in_view(camera, {0.0f, 0.0f, 2.0f}, 0.01));
  // Far past the right and the top: the half extents a unit ahead are
  // tan 20 degrees times the aspect, and tan 20 degrees.
  CHECK_FALSE(in_view(camera, {3.0f, 0.0f, -2.0f}, 0.01));
  CHECK_FALSE(in_view(camera, {0.0f, -3.0f, -2.0f}, 0.01));
  // Its center past the left side by less than its radius: kept.
  const float edge = 2.0f * 0.36397f * (400.0f / 300.0f);  // x at depth 2
  CHECK(in_view(camera, {-(edge + 0.05f), 0.0f, -2.0f}, 0.1));
  CHECK_FALSE(in_view(camera, {-(edge + 0.5f), 0.0f, -2.0f}, 0.1));
}

}  // namespace serenity::shapes
