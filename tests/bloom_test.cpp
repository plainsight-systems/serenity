// The bloom pyramid's level sizes (core/passes/bloom.h): each the one before
// halved on each axis, rounded up, at least 1, so a frame of any size has
// every level. The tone-map pass's GPU test reads the levels the shader made
// against its own reference (tests/gpu/tone_map_test.cpp).

#include "core/passes/bloom.h"

#include <doctest/doctest.h>

#include "core/frame/extent.h"
#include "core/passes/tone_map.h"

namespace serenity::passes {
namespace {

TEST_CASE("a level is the one before halved, rounded up, at least 1") {
  CHECK(bloom_level({1, 1}) == frame::Extent{1, 1});
  CHECK(bloom_level({2, 1}) == frame::Extent{1, 1});
  CHECK(bloom_level({1, 3}) == frame::Extent{1, 2});
  CHECK(bloom_level({5, 3}) == frame::Extent{3, 2});
  CHECK(bloom_level({1920, 1080}) == frame::Extent{960, 540});
}

TEST_CASE("a frame of one pixel has every level, each one pixel") {
  frame::Extent level{1, 1};
  for (std::uint32_t k = 0; k < bloom_levels; ++k) {
    level = bloom_level(level);
    CHECK(level == frame::Extent{1, 1});
  }
}

}  // namespace
}  // namespace serenity::passes
