// The grid a light runs at on the Neural Engine (metal/passes/denoise/
// rungs.h), a unit with no Metal call: the smallest of the model's shapes
// that holds the light's padded grid, by pixels, the narrower of two the
// same; none past the largest.

#include "metal/passes/denoise/rungs.h"

#include <doctest/doctest.h>

#include <array>

namespace serenity::metal {
namespace {

TEST_CASE("a light runs at the smallest rung that holds its padded grid") {
  // Rungs of no network in particular, out of order, two of one area.
  const std::array<frame::Extent, 5> rungs{
      {{960, 544}, {256, 256}, {1728, 1120}, {544, 960}, {2048, 2048}}};
  // A rung exactly, after padding: 960 x 540 pads to 960 x 544.
  CHECK(smallest_rung(rungs, {960, 540}) == frame::Extent{960, 544});
  CHECK(smallest_rung(rungs, {1728, 1117}) == frame::Extent{1728, 1120});
  // Between rungs: the smallest by pixels that holds it on both axes.
  CHECK(smallest_rung(rungs, {100, 300}) == frame::Extent{544, 960});
  CHECK(smallest_rung(rungs, {900, 900}) == frame::Extent{1728, 1120});
  // Two rungs of one area both hold it: the narrower.
  CHECK(smallest_rung(rungs, {500, 500}) == frame::Extent{544, 960});
  // Padding decides: 1 past a rung's side is the next rung.
  CHECK(smallest_rung(rungs, {257, 200}) == frame::Extent{544, 960});
  // Past the largest: none.
  CHECK_FALSE(smallest_rung(rungs, {2049, 16}).has_value());
  CHECK_FALSE(smallest_rung({}, {16, 16}).has_value());
}

}  // namespace
}  // namespace serenity::metal
