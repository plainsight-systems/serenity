// The size rule for an image Metal makes (metal/device/device.h,
// check_texture_size), a unit with no Metal call: each side from 1 to
// max_texture_side, refused otherwise by Error naming the image, not by
// Metal stopping the program. That the largest it allows is one this GPU
// makes is frame_errors_test.cpp's, on the device.

#include <doctest/doctest.h>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "support/text.h"

namespace serenity::metal {
namespace {

TEST_CASE("an image's sides are each from 1 to the largest Metal makes") {
  CHECK_NOTHROW(check_texture_size({1, 1}, "image"));
  CHECK_NOTHROW(
      check_texture_size({max_texture_side, max_texture_side}, "image"));
  CHECK_THROWS_AS(check_texture_size({max_texture_side + 1, 16}, "image"),
                  MetalError);
  CHECK_THROWS_AS(check_texture_size({16, max_texture_side + 1}, "image"),
                  MetalError);
  CHECK_THROWS_AS(check_texture_size({0, 16}, "image"), MetalError);
  try {
    check_texture_size({16, 0}, "the radiance image");
    FAIL("an image of no rows was allowed");
  } catch (const MetalError& refused) {
    CHECK(tests::contains(refused.what(), "the radiance image"));
  }
}

}  // namespace
}  // namespace serenity::metal
