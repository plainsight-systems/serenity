// The camera's jitter (core/sampling/jitter.h): Halton(2, 3) of 32 phases,
// pinned to the table training/test_training.py and
// tests/test_make_denoiser_dataset.py hold their own copies of the formula
// to; its 32 phases cover the pixel; and which graphs take it
// (core/frame/schedule.h, jitters).

#include "core/sampling/jitter.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>

#include "core/frame/schedule.h"
#include "support/schedules.h"

namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace sampling = serenity::sampling;
namespace tests = serenity::tests;

TEST_CASE(
    "the camera's jitter is Halton(2, 3) from k = 1, its 32 phases repeating, "
    "the same as its copies'") {
  // index, x, y: the table the Python copies are held to.
  constexpr std::array<std::pair<std::uint64_t, std::array<double, 2>>, 7>
      table = {{
          {0, {0.0, -1.0 / 6.0}},
          {1, {-0.25, 1.0 / 6.0}},
          {2, {0.25, -7.0 / 18.0}},
          {31, {-0.484375, 0.2901234567901234}},
          {32, {0.0, -1.0 / 6.0}},
          {7200, {0.0, -1.0 / 6.0}},
          {7201, {-0.25, 1.0 / 6.0}},
      }};
  for (const auto& [index, expected] : table) {
    const sampling::CameraJitter j = sampling::camera_jitter(index);
    INFO("frame " << index);
    CHECK(j.x == doctest::Approx(expected[0]).epsilon(1e-7));
    CHECK(j.y == doctest::Approx(expected[1]).epsilon(1e-7));
  }
  static_assert(sampling::camera_jitter(5).x ==
                sampling::camera_jitter(5 + contracts::jitter_phases).x);
}

TEST_CASE(
    "the jitter's 32 phases cover the pixel: each x in its own 1/32, each of "
    "the first 27 y in its own 1/27") {
  std::set<int> x_strata;
  std::set<int> y_strata;
  for (std::uint64_t i = 0; i < contracts::jitter_phases; ++i) {
    const sampling::CameraJitter j = sampling::camera_jitter(i);
    CHECK(j.x >= -0.5f);
    CHECK(j.x < 0.5f);
    CHECK(j.y >= -0.5f);
    CHECK(j.y < 0.5f);
    x_strata.insert(static_cast<int>((static_cast<double>(j.x) + 0.5) * 32.0));
    if (i < 27) {
      y_strata.insert(
          static_cast<int>((static_cast<double>(j.y) + 0.5) * 27.0 +
                           1e-4));  // m / 27 in float, a little under
    }
  }
  CHECK(x_strata.size() == contracts::jitter_phases);
  CHECK(y_strata.size() == 27);
}

TEST_CASE(
    "the graphs whose light is the network's input take the jitter; the path "
    "tracer and the features alone "
    "do not") {
  using frame::PassKind;
  const auto restir = [](bool features) {
    std::vector<PassKind> light = tests::restir_pt_passes;
    if (features) {
      light.push_back(PassKind::features);
    }
    light.push_back(PassKind::tone_map);
    frame::Schedule s{light, tests::look};
    s.restir_pt = tests::written_restir_pt(light);
    return s;
  };
  const std::optional<sampling::CameraJitter> taken =
      frame::camera_jitter(restir(true), 7);
  REQUIRE(taken.has_value());
  CHECK(taken->x == sampling::camera_jitter(7).x);
  CHECK(taken->y == sampling::camera_jitter(7).y);
  CHECK_FALSE(frame::camera_jitter(restir(false), 7).has_value());
  CHECK_FALSE(frame::camera_jitter(
      frame::Schedule{{PassKind::path, PassKind::tone_map}, tests::look}, 7));
  CHECK_FALSE(frame::camera_jitter(
      frame::Schedule{{PassKind::features}, std::nullopt}, 7));
}
