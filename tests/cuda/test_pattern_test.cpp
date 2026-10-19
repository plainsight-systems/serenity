// The CUDA test pattern (cuda/passes/test_pattern/test_pattern.h) against
// the CPU's: red and green the same to the bit (IEEE division, no fast
// math: cmake/CudaModules.cmake), blue within one level (the GPU's sinf and
// the CPU's sin may differ in the last bit), alpha opaque; the same frame
// rendered twice the same bytes; and the backend refusing, by name, a pass
// it does not build yet.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/device/device.h"
#include "cuda/device/error.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"
#include "support/scene_text.h"
#include "support/schedules.h"
#include "test_pattern_reference.h"

namespace serenity::cuda {
namespace {

const frame::Schedule pattern{{frame::PassKind::test_pattern}, std::nullopt};

// One frame of the pattern at `seconds`, `size`, read back.
std::vector<std::uint8_t> rendered(frame::Extent size, double seconds) {
  const Device device;
  Submission submission(device);
  Offscreen target(submission, size);
  Renderer renderer(submission, pattern, nullptr);
  (void)render_to_offscreen(
      submission, target, renderer,
      frame::FrameInputs{
          .time = frame::Seconds(seconds), .index = 0, .camera = std::nullopt});
  std::vector<std::uint8_t> rgba(target.rgba_size());
  target.read_rgba(rgba);
  return rgba;
}

TEST_CASE("the test pattern is the CPU's, blue within a level") {
  // Not a multiple of the 16 x 16 blocks, so the edges' threads return.
  const frame::Extent size{37, 23};
  for (const double seconds : {0.0, 0.3, 1.0, 2.7}) {
    CAPTURE(seconds);
    const std::vector<std::uint8_t> rgba = rendered(size, seconds);
    int worst_blue = 0;
    for (unsigned y = 0; y < size.height; ++y) {
      for (unsigned x = 0; x < size.width; ++x) {
        const std::size_t at = (std::size_t{y} * size.width + x) * 4;
        const std::array<std::uint8_t, 4> want = tests::test_pattern_pixel(
            x, y, size.width, size.height, static_cast<float>(seconds));
        CHECK(rgba.at(at) == want[0]);
        CHECK(rgba.at(at + 1) == want[1]);
        worst_blue = std::max(worst_blue, std::abs(rgba.at(at + 2) - want[2]));
        CHECK(rgba.at(at + 3) == 255);
      }
    }
    CHECK(worst_blue <= 1);
  }
}

TEST_CASE("the same frame renders the same bytes") {
  CHECK(rendered({64, 32}, 0.3) == rendered({64, 32}, 0.3));
}

TEST_CASE("a pass the CUDA backend does not build yet is refused by name") {
  const Device device;
  Submission submission(device);
  const frame::Schedule path{{frame::PassKind::path, frame::PassKind::tone_map},
                             tests::look};
  const scene::SceneDescription scene =
      scene::parse(tests::flying_fireflies(), "flying fireflies");
  try {
    Renderer renderer(submission, path, &scene);
    FAIL("a graph with a pass the backend does not build was accepted");
  } catch (const CudaError& refused) {
    CHECK(std::string(refused.what()).find("has no path pass yet") !=
          std::string::npos);
  }
}

}  // namespace
}  // namespace serenity::cuda
