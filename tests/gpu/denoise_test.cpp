// The denoised pipeline, two integration tests. The denoise pass
// (passes/denoise/denoise.h) against PyTorch: the
// committed fixture, tests/data/denoise (training/denoise_check.py made it:
// a network of training/model.py's structure at small widths, randomly
// initialized, head included; three consecutive frames of a training
// sequence's renderer images, a 40 x 22 crop where fireflies move, so the
// U-Net's padding, the masked taps, the depth median, the motion vectors
// and the history all take part; and model.Denoiser.frame's float32
// outputs). Run through the GPU as the renderer runs it, frame after frame
// with its own history:
//
//   - each output is PyTorch's to within `tolerance`, relative: half
//     floats in the U-Net, float sums, on either backend (Metal's
//     simdgroup matrices, denoise.metal, "Precision"; CUDA's tensor cores,
//     cuda/passes/denoise/denoise.h). On this fixture Metal's largest is
//     2.2e-4 (docs/research/2026-10-15-denoiser-live.md), so 1e-3 leaves a
//     margin of four;
//   - and the check can fail: a network one weight off is not, one of the
//     last convolution's weights a tenth off giving 4.0e-3, four times the
//     tolerance.
//
// The pipeline in the renderer, ReSTIR PT's light traced checkerboarded,
// filled, denoised and tone mapped (gpu/support/rendering.h,
// fixture_denoised_graph), over moving fireflies, two queues against one:
// frame N + 1's light half never writes what frame N's image half reads,
// every value finite and the scene shown, and the two runs' same bits the
// pipeline's determinism (below).

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/output/pfm.h"
#include "core/scene/scene.h"
#include "gpu/support/denoise_fixture.h"
#include "gpu/support/rendering.h"
#include "support/same_bits.h"
#include "support/scene_text.h"
#include "test_paths.h"

// Each backend's runner and hand-off rig, by the include path (SF.12).
#include <denoise_runner.h>
#include <hand_off_rig.h>

namespace contracts = serenity::contracts;
namespace denoise = serenity::denoise;
namespace frame = serenity::frame;
namespace output = serenity::output;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

namespace {

constexpr double tolerance = 1e-3;
// Below this, a value counts as this: the floor of the relative test, so
// near-black pixels are held to an absolute 1e-6.
constexpr double floor_value = 1e-3;

// The largest |gpu - expected| / (|expected| + floor_value), over every
// frame, pixel and channel.
double worst(const tests::DenoiseFixture& fixture,
             const std::vector<contracts::LinearImage>& outputs) {
  REQUIRE(outputs.size() == fixture.frames);
  double largest = 0.0;
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    const contracts::LinearImage expected =
        output::read_pfm(fixture.file(k, "expected"));
    REQUIRE(expected.extent == outputs[k].extent);
    for (std::size_t i = 0; i < expected.rgb.size(); ++i) {
      const double e = expected.rgb[i];
      const double g = outputs[k].rgb[i];
      REQUIRE(std::isfinite(g));
      largest =
          std::max(largest, std::abs(g - e) / (std::abs(e) + floor_value));
    }
  }
  return largest;
}

}  // namespace

TEST_CASE(
    "the denoise pass computes what PyTorch computes, frame after frame") {
  const tests::DenoiseFixture fixture =
      tests::read_denoise_fixture(tests::data_dir / "denoise");
  REQUIRE(fixture.frames == 3);
  const double error = worst(fixture, tests::run_denoise_fixture(fixture));
  INFO("largest relative difference: " << error);
  CHECK(error < tolerance);
  // And the check can fail: a network one weight off, one weight of the
  // head by a tenth (the 1 x 1 each backend computes itself; Metal's U-Net
  // is its package, compiled from the weights on disk), does not pass it.
  const double off = worst(fixture, tests::run_denoise_fixture(
                                        fixture, [](denoise::Network& network) {
                                          network.head.weight.at(0) += 0.1f;
                                        }));
  INFO("one weight off: largest relative difference " << off);
  CHECK(off > tolerance);
}

// The hand-off (metal/frame/renderer.h, "Two halves"; cuda/frame/
// renderer.h, "A denoised frame"): frame N + 1's light half never writes
// what frame N's image half reads. Frames with their halves on two queues
// (two streams on CUDA), two in flight and none waited for, end in the same
// bits as the same frames each recorded into one part and waited for: every
// frame as displayed, and the last frame's radiance. A hand-off frame N + 1
// overwrote under frame N's image half would change frame N's image (the
// full-size albedo, which the denoiser multiplies back after its history)
// or reach the denoiser's history and the frames after it (the radiance and
// the features).
//
// Left to the GPU, frame N's image half reads the hand-off at its start,
// before frame N + 1's light half has reached its end and written it, so a
// shared hand-off would pass too. So each frame's image half is held back:
// its second part begins with the tests' delay kernel (each backend's
// <hand_off_rig.h>, some 30 ms), which the image half waits for, while
// frame N + 1's light half, a few milliseconds here, runs to its end
// beside it. The test shows it was so: each frame's light half began
// before the frame before it had finished (the records, each backend's
// submission's), and every frame after the second was held back for longer
// than the frame before it took to trace its light (by the records too: the
// image half's start, which the delay sets, is after the next frame's light
// half ended, within the GPU's own times).

namespace {

// Frame `index` of the hand-off's run, 50 ms after the one before, so the
// fireflies move a pixel or so between frames: every image the halves hand
// over, the full-size albedo's among them, differs from frame to frame.
frame::FrameInputs stepped_frame(std::uint64_t index,
                                 const scene::SceneDescription& scene) {
  const auto at = [](std::uint64_t i) {
    return frame::Seconds{30.0 + 0.05 * static_cast<double>(i)};
  };
  frame::FrameInputs inputs =
      tests::frame_at(index, index, at(index).count(), scene.camera);
  inputs.reused_since = 0;
  if (index > 0) {
    inputs.previous = frame::PreviousFrame{at(index - 1), scene.camera};
  }
  return inputs;
}

// The hand-off run's frames: `frames` of them, 50 ms apart.
std::vector<frame::FrameInputs> stepped_frames(
    std::uint64_t frames, const scene::SceneDescription& scene) {
  std::vector<frame::FrameInputs> out;
  out.reserve(frames);
  for (std::uint64_t i = 0; i < frames; ++i) {
    out.push_back(stepped_frame(i, scene));
  }
  return out;
}

}  // namespace

TEST_CASE(
    "frame N + 1's light half never writes what frame N's image half reads") {
  const scene::SceneDescription scene =
      scene::parse(tests::wandering_fireflies(), "wandering fireflies");
  constexpr frame::Extent size{256, 192};
  constexpr std::uint64_t frames = 6;
  const std::vector<frame::FrameInputs> stepped = stepped_frames(frames, scene);
  const tests::HandOffRun overlapped =
      tests::run_hand_off(scene, size, stepped, tests::Halves::overlapped);
  const tests::HandOffRun serial =
      tests::run_hand_off(scene, size, stepped, tests::Halves::serial);
  REQUIRE(overlapped.shown.size() == frames);
  for (std::size_t k = 0; k < frames; ++k) {
    INFO("frame " << k);
    CHECK(overlapped.shown[k] == serial.shown[k]);
  }
  CHECK(tests::same_bits(overlapped.last, serial.last));
  // The pipeline runs, every value finite, and it shows the scene: light,
  // not a black or an empty image. Two runs ending in the same bits, as
  // these do, is also the pipeline's determinism.
  CHECK(std::ranges::all_of(serial.last,
                            [](float v) { return std::isfinite(v); }));
  CHECK(std::ranges::any_of(serial.last, [](float v) { return v > 0.0f; }));

  // It exercised the hazard: each frame's light half (its record's start)
  // began, and ended, while the frame before was held back. A serial
  // frame's span is its light half and image half alone, no delay: the
  // longest of them bounds a light half, and each overlapped frame's own
  // span is far longer, the delay's. Both from the second frame on: on
  // CUDA, the first frame waits for the device once, as OptiX makes ready
  // for its first launches (measured on the RTX 3090 Ti: the path tree's
  // first launch 11 ms, MNEE's 4, every later one 0.01), which bounds no
  // light half, and holds the overlapped run's second frame back until the
  // first ends; each frame from the third is the claim's.
  REQUIRE(overlapped.records.size() == frames);
  REQUIRE(serial.records.size() == frames);
  double longest_serial = 0.0;
  for (std::size_t k = 1; k < serial.records.size(); ++k) {
    const tests::FrameSpan& record = serial.records[k];
    longest_serial =
        std::max(longest_serial, (record.gpu_end - record.gpu_start).count());
  }
  for (std::size_t k = 2; k < overlapped.records.size(); ++k) {
    const tests::FrameSpan& before = overlapped.records[k - 1];
    const tests::FrameSpan& next = overlapped.records[k];
    INFO("frame "
         << k << ": began " << (before.gpu_end - next.gpu_start).count() * 1e3
         << " ms before the frame before ended; a serial frame takes at most "
         << longest_serial * 1e3 << " ms");
    CHECK(next.gpu_start + frame::Seconds{longest_serial} < before.gpu_end);
  }
}
