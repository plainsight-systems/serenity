#pragma once

// Frames rendered the way the GPU tests render them (ES.3, F.10): the frame
// graphs they run, a frame's inputs, a patch's mean, and how two images
// differ; a frame rendered whole on a renderer of its own and read back,
// and a renderer whose frames accumulate, are each backend's own
// (accumulating_rig.h: Metal's beside this file, CUDA's in tests/cuda/
// support/), under one name, so a test of whole frames is one source for
// both.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// the backend's renders, by the include path (SF.12): its AccumulatingRig
// and read_back()
#include <accumulating_rig.h>

#include "core/frame/frame_inputs.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "support/files.h"
#include "support/references.h"
#include "support/schedules.h"
#include "support/text.h"
#include "test_paths.h"

namespace serenity::tests {

// The path tracer, shown through the tone map, no exposure and no bloom;
// what the tests read is its accumulated image, at full precision.
inline frame::Schedule path_graph() {
  return frame::parse_schedule(
      "passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = 0.0\nbloom = "
      "0.0\n",
      "path graph");
}

// The denoised pipeline: ReSTIR PT's passes, the features, the denoise
// pass, the depth of field and the tone map, ReSTIR PT's settings the written
// ones (support/schedules.h), its network the test fixture's
// (tests/data/denoise/network), not the one beside the checkout.
inline frame::Schedule fixture_denoised_graph() {
  std::vector<frame::PassKind> kinds = restir_pt_passes;
  kinds.push_back(frame::PassKind::features);
  kinds.push_back(frame::PassKind::denoise);
  kinds.push_back(frame::PassKind::depth_of_field);
  kinds.push_back(frame::PassKind::tone_map);
  frame::Schedule s{kinds, look};
  s.restir_pt = written_restir_pt(kinds);
  s.denoise = passes::Denoise{data_dir / "denoise" / "network"};
  return s;
}

// Frame `index` at `seconds`, its image accumulated since frame `since`,
// seen by `camera`.
inline frame::FrameInputs frame_at(std::uint64_t index, std::uint64_t since,
                                   double seconds,
                                   std::optional<contracts::Camera> camera) {
  return frame::FrameInputs{.time = frame::Seconds(seconds),
                            .index = index,
                            .accumulated_since = since,
                            .camera = camera};
}

// A square of pixels: its middle, and how far it reaches either way.
struct Patch {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t half = 0;
};

// The square of half-width `half` in the middle of an image of `size`.
constexpr Patch middle_of(frame::Extent size, std::uint32_t half) noexcept {
  return {size.width / 2, size.height / 2, half};
}

// The mean of `channel` over `patch` of an RGBA image of `size`, each
// value read by `value`. The patch must lie within the image: one at an
// edge is a failed test, not an index wrapped past it (ES.106, SL.con.3).
template <typename T, typename Value>
double patch_mean(std::span<const T> rgba, frame::Extent size, Patch patch,
                  std::size_t channel, Value value) {
  REQUIRE(rgba.size() == std::size_t{size.width} * size.height * 4);
  REQUIRE(channel < 4);
  REQUIRE(patch.x >= patch.half);
  REQUIRE(patch.y >= patch.half);
  REQUIRE(patch.x + patch.half < size.width);
  REQUIRE(patch.y + patch.half < size.height);
  double sum = 0.0;
  for (std::uint32_t y = patch.y - patch.half; y <= patch.y + patch.half; ++y) {
    for (std::uint32_t x = patch.x - patch.half; x <= patch.x + patch.half;
         ++x) {
      sum += value(rgba[(std::size_t{y} * size.width + x) * 4 + channel]);
    }
  }
  const std::uint32_t side = 2 * patch.half + 1;
  return sum / (side * side);
}

// A displayed image's: its 8-bit sRGB values decoded to linear radiance.
inline double patch_mean(std::span<const std::uint8_t> rgba, frame::Extent size,
                         Patch patch, std::size_t channel = 0) {
  return patch_mean(rgba, size, patch, channel,
                    [](std::uint8_t v) { return linear_of(v); });
}

// An accumulated image's: its linear radiance as it is.
inline double patch_mean(std::span<const float> rgba, frame::Extent size,
                         Patch patch, std::size_t channel = 0) {
  return patch_mean(rgba, size, patch, channel,
                    [](float v) { return static_cast<double>(v); });
}

// How many bytes two images of one size differ in: 0 for the same image.
// A count, not the images, is what a failing check prints.
inline std::size_t differing(const std::vector<std::uint8_t>& a,
                             const std::vector<std::uint8_t>& b) {
  REQUIRE(a.size() == b.size());
  return std::transform_reduce(a.begin(), a.end(), b.begin(), std::size_t{0},
                               std::plus<>(),
                               [](std::uint8_t x, std::uint8_t y) {
                                 return std::size_t{x != y ? 1u : 0u};
                               });
}

}  // namespace serenity::tests
