// The frame's submission (metal/device/submission.h), an integration test:
// frames in flight each reading their own frame's constants from the ring
// (metal/frame/renderer.h), the test pattern rendered into images of their
// own and read back. The submission's protocol, which submission takes which
// slot, what is settled once, and the misuses refused, is a unit apart from
// Metal, tested without a GPU (frame_sequence_test.cpp).

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <vector>

#include "core/frame/graph_file.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/references.h"

namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace tests = serenity::tests;

namespace {

constexpr frame::Extent size{64, 32};

// The test pattern's blue pulses once every this many seconds
// (metal/passes/test_pattern/test_pattern.h).
constexpr double pulse_period = 4.0;

frame::Schedule test_pattern_schedule() {
  return frame::parse_schedule("passes = [\"test_pattern\"]\n", "test");
}

frame::FrameInputs at(double seconds, std::uint64_t index) {
  return tests::frame_at(index, index, seconds, std::nullopt);
}

// The test pattern's formula (test_pattern.h), as 8-bit values.
int expected(double value) {
  return static_cast<int>(std::lround(value * 255.0));
}

void check_pixel(const std::vector<std::uint8_t>& rgba, tests::Pixel pixel,
                 double seconds) {
  const std::size_t at = (std::size_t{pixel.y} * size.width + pixel.x) * 4;
  const double red = static_cast<double>(pixel.x) / (size.width - 1);
  const double green = static_cast<double>(pixel.y) / (size.height - 1);
  const double blue =
      0.5 + 0.5 * std::sin(2.0 * std::numbers::pi * seconds / pulse_period);
  INFO("pixel (" << pixel.x << ", " << pixel.y << ") at t = " << seconds);
  // Within one step of 8-bit storage and of the GPU's sin.
  CHECK(std::abs(rgba[at + 0] - expected(red)) <= 1);
  CHECK(std::abs(rgba[at + 1] - expected(green)) <= 1);
  CHECK(std::abs(rgba[at + 2] - expected(blue)) <= 1);
  CHECK(rgba[at + 3] == 255);
}

// The corners and a pixel inside.
constexpr std::array<tests::Pixel, 5> probed{{{0, 0},
                                              {size.width - 1, 0},
                                              {0, size.height - 1},
                                              {size.width - 1, size.height - 1},
                                              {21, 13}}};

}  // namespace

TEST_CASE("frames in flight: each submission reads its own frame's constants") {
  // Four frames, each into its own image, at four times, committed without
  // waiting, so two are in flight at once and the ring's two slots are each
  // used twice. Had the frames shared one constants slot, an earlier frame
  // would show a later frame's time.
  const std::array<double, 4> times{
      1.0, 3.0, 0.5, 2.5};  // blue 1, 0, ~0.85, ~0.15: all distinct
  metal::Device device;
  metal::Submission submission(device);
  metal::Renderer renderer(device, submission, test_pattern_schedule(),
                           nullptr);
  std::vector<std::unique_ptr<metal::Offscreen>> targets;
  std::uint64_t last = 0;
  for (std::size_t i = 0; i < times.size(); ++i) {
    targets.push_back(
        std::make_unique<metal::Offscreen>(device, submission, size));
    last = metal::render_to_offscreen(submission, *targets.back(), renderer,
                                      at(times.at(i), i));
  }
  (void)submission.wait_until_complete(
      last);  // in order, so every earlier one is done too
  for (std::size_t i = 0; i < times.size(); ++i) {
    const std::vector<std::uint8_t> rgba = tests::read_back(*targets[i]);
    check_pixel(rgba, probed[3], times.at(i));
    check_pixel(rgba, probed[4], times.at(i));
  }
}
