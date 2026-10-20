#pragma once

// CUDA's run of the hand-off's frames (tests/gpu/denoise_test.cpp, "frame
// N + 1's light half never writes what frame N's image half reads"),
// Metal's (tests/gpu/support/hand_off_rig.h) under the same names and
// interface: frames of the fixture's denoised graph (gpu/support/
// rendering.h), each into a target of its own, so every frame's image is
// read back, not only the last's: of two parts, each image half held back
// by the tests' delay kernel (tests/cuda/kernels/delay.cu), put on the
// image stream before the frame is recorded (cuda/device/submission.h,
// image_stream), so the image half waits for it, none waited for; or of
// one part, each waited for. With every frame's GPU instants
// (submission.h, "Instants").

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/scene/scene.h"
#include "cuda/device/buffer.h"
#include "cuda/device/device.h"
#include "cuda/device/error.h"
#include "cuda/device/module.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"
#include "gpu/support/rendering.h"
#include "serenity/cuda_modules/probes.h"

namespace serenity::tests {

// How a hand-off run's frames are begun: in two parts, overlapped, each
// image half held back; or in one, each waited for.
enum class Halves : std::uint8_t { overlapped, serial };

// A frame's GPU instants, on the backend's one clock.
struct FrameSpan {
  std::uint64_t sequence = 0;
  frame::Seconds gpu_start{0.0};
  frame::Seconds gpu_end{0.0};
};

struct HandOffRun {
  std::vector<std::vector<std::uint8_t>> shown;  // each frame, as displayed
  std::vector<float> last;         // the last frame's denoised radiance
  std::vector<FrameSpan> records;  // the frames', in sequence order
};

namespace hand_off {

// The delay and its count of steps: some 20 ms on the RTX 3090 Ti, as
// Metal's 10 million on the M3 Max.
class Delay {
 public:
  Delay()
      : function_(module_.kernel("delay")),
        out_(sizeof(std::uint32_t)),
        count_(sizeof(std::uint32_t)) {
    constexpr std::uint32_t steps = 10'000'000;
    cuda::upload_now(count_.address(), std::as_bytes(std::span(&steps, 1)),
                     "Delay");
  }

  // One thread on `stream`.
  void record(CUstream stream) {
    CUdeviceptr out = out_.address();
    CUdeviceptr count = count_.address();
    std::array<void*, 2> arguments{&out, &count};
    cuda::launch(stream, function_, cuda::LaunchShape{}, arguments);
  }

 private:
  cuda::Module module_{cuda_modules::probes};
  CUfunction function_;
  cuda::DeviceBuffer out_;
  cuda::DeviceBuffer count_;
};

}  // namespace hand_off

// `frames` frames through the fixture's denoised graph at `size` over
// `scene` (above), begun as `halves` says.
inline HandOffRun run_hand_off(const scene::SceneDescription& scene,
                               frame::Extent size,
                               std::span<const frame::FrameInputs> frames,
                               Halves halves) {
  const cuda::Parts parts =
      halves == Halves::overlapped ? cuda::Parts::two : cuda::Parts::one;
  cuda::Device device;
  cuda::Submission submission(device);
  hand_off::Delay delay;
  cuda::Renderer renderer(device, submission, tests::fixture_denoised_graph(),
                          &scene);
  REQUIRE(renderer.parts() == cuda::Parts::two);
  std::vector<std::unique_ptr<cuda::Offscreen>> targets;
  targets.reserve(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    targets.push_back(std::make_unique<cuda::Offscreen>(submission, size));
  }
  // The frames' sequences start here: the submissions before them, the
  // structures' build, are no frames of the run.
  const std::uint64_t first = submission.next_sequence();
  HandOffRun run;
  std::vector<cuda::Completed> completed;
  const auto keep = [&] {
    submission.take_completed(completed);
    for (const cuda::Completed& c : completed) {
      if (c.sequence >= first) {
        run.records.push_back(FrameSpan{c.sequence, c.gpu_start, c.gpu_end});
      }
    }
    completed.clear();
  };
  // The renderer records a frame in its graph's parts; a serial run
  // records each frame's two halves on the light stream, the same work in
  // one part, as Metal's serial run does.
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const frame::FrameInputs& inputs = frames[i];
    renderer.prepare(inputs, size);
    const cuda::FrameSlot begun = submission.begin(parts);
    if (parts == cuda::Parts::two) {
      delay.record(submission.image_stream());
    }
    renderer.record(begun, inputs, *targets[i]);
    submission.commit();
    if (parts == cuda::Parts::one) {
      submission.wait_until_complete(begun.sequence);
    }
    keep();
  }
  submission.finish();
  keep();
  std::ranges::sort(run.records, {}, &FrameSpan::sequence);
  for (const std::unique_ptr<cuda::Offscreen>& target : targets) {
    run.shown.push_back(tests::read_back(*target));
  }
  run.last.assign(std::size_t{size.width} * size.height * 4, 0.0f);
  renderer.read_accumulated(run.last);
  return run;
}

}  // namespace serenity::tests
