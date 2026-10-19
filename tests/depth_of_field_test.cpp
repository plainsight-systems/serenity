// The depth of field's constants (core/passes/depth_of_field.h, contract
// 21): the half grid covers the output, the tiles cover the grid, a tile's
// dilation reaches as far as the largest radius can spread, and a pinhole
// blurs nothing. That the scale is the traced lens's blur is the GPU test's
// (tests/gpu/camera_test.cpp).

#include "core/contracts/depth_of_field.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>

#include "core/camera/thin_lens.h"
#include "core/passes/depth_of_field.h"

using namespace serenity;

namespace {

contracts::Camera lensed(float radius) {
  contracts::Camera c{
      {0.0f, 0.88f, 0.3f}, {0.0f, 0.757f, -0.07f}, {0.0f, 1.0f, 0.0f}, 28.0f};
  c.lens_radius = radius;
  c.focus_distance = 0.39f;
  return c;
}

}  // namespace

TEST_CASE(
    "the depth of field's grid, tiles and dilation cover the output and its "
    "largest blur") {
  for (const frame::Extent size :
       {frame::Extent{3456, 2234}, frame::Extent{1920, 1080},
        frame::Extent{1921, 3}, frame::Extent{1, 1}}) {
    const contracts::DepthOfFieldConstants k =
        passes::depth_of_field_constants(lensed(0.003f), size);
    INFO(size.width << " x " << size.height);
    CHECK(k.width == size.width);
    CHECK(k.height == size.height);
    // Every output pixel in a half pixel, and no half pixel outside.
    CHECK(k.half_width * contracts::dof_scale >= size.width);
    CHECK((k.half_width - 1u) * contracts::dof_scale < size.width);
    CHECK(k.half_height * contracts::dof_scale >= size.height);
    CHECK((k.half_height - 1u) * contracts::dof_scale < size.height);
    CHECK(k.tiles_x * contracts::dof_tile >= k.half_width);
    CHECK((k.tiles_x - 1u) * contracts::dof_tile < k.half_width);
    CHECK(k.tiles_y * contracts::dof_tile >= k.half_height);
    // The dilation reaches as far as the largest radius, in half pixels.
    CHECK(static_cast<float>(k.dilation * contracts::dof_tile) >=
          k.max_radius / contracts::dof_scale);
    CHECK(k.dilation >= 1u);
    CHECK(k.max_radius ==
          contracts::dof_max_radius * static_cast<float>(size.width));
    CHECK(passes::blurs(k));
    CHECK(k.coc_scale ==
          camera::circle_of_confusion_scale(lensed(0.003f), size));
  }
}

TEST_CASE(
    "a pinhole blurs nothing, and a camera that cannot be framed is refused") {
  CHECK_FALSE(passes::blurs(
      passes::depth_of_field_constants(lensed(0.0f), {1920, 1080})));
  contracts::Camera broken = lensed(0.003f);
  broken.look_at = broken.position;
  CHECK_THROWS_AS(passes::depth_of_field_constants(broken, {1920, 1080}),
                  std::invalid_argument);
  CHECK_THROWS_AS(passes::depth_of_field_constants(lensed(0.003f), {0, 1080}),
                  std::invalid_argument);
}
