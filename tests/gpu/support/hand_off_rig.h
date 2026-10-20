#pragma once

// Metal's run of the hand-off's frames (tests/gpu/denoise_test.cpp, "frame
// N + 1's light half never writes what frame N's image half reads"): frames
// of the fixture's denoised graph (gpu/support/rendering.h), each into a
// target of its own, so every frame's image is read back, not only the
// last's: of two parts, each image half held back by the tests' delay
// kernel (kernels/delay.metal, about 30 ms on the M3 Max), which the image
// half's queue barrier waits for, none waited for; or of one part, each
// waited for. With every frame's GPU instants (metal/device/submission.h).
// CUDA's is tests/cuda/support/hand_off_rig.h, under the same names and
// interface.

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/scene/scene.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/device/support.h"
#include "metal/frame/renderer.h"
#include "serenity/metallib/probes.h"

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

// The delay, its count of steps (kernels/delay.metal) and its buffers,
// recorded at the start of a second part.
class Delay {
 public:
  Delay(const metal::Device& device, metal::Submission& submission)
      : library_(device, metallib::probes),
        pipeline_(library_.compute_pipeline("delay")) {
    const auto pool = metal::scoped_pool();
    MTL::Device* mtl = device.handle();
    out_ = NS::TransferPtr(
        mtl->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    count_ = NS::TransferPtr(
        mtl->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    REQUIRE(out_);
    REQUIRE(count_);
    constexpr std::uint32_t steps = 10'000'000;
    std::memcpy(count_->contents(), &steps, sizeof steps);
    submission.make_resident(out_.get());
    submission.make_resident(count_.get());
    auto descriptor =
        NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(2);
    NS::Error* error = nullptr;
    table_ = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    INFO("argument table: " << reason(error));
    REQUIRE(table_);
    table_->setAddress(out_->gpuAddress(), 0);
    table_->setAddress(count_->gpuAddress(), 1);
  }

  void record(MTL4::CommandBuffer* commands) const {
    MTL4::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    encoder->setArgumentTable(table_.get());
    encoder->setComputePipelineState(pipeline_.get());
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->endEncoding();
  }

 private:
  metal::Library library_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  NS::SharedPtr<MTL::Buffer> out_;
  NS::SharedPtr<MTL::Buffer> count_;
  NS::SharedPtr<MTL4::ArgumentTable> table_;
};

inline FrameSpan span(const metal::Completed& c) {
  return FrameSpan{c.sequence, c.gpu_start, c.gpu_end};
}

}  // namespace hand_off

// `frames` frames through the fixture's denoised graph at `size` over
// `scene` (above), begun as `halves` says.
inline HandOffRun run_hand_off(const scene::SceneDescription& scene,
                               frame::Extent size,
                               std::span<const frame::FrameInputs> frames,
                               Halves halves) {
  using hand_off::Delay;
  using hand_off::span;
  const metal::Parts parts =
      halves == Halves::overlapped ? metal::Parts::two : metal::Parts::one;
  metal::Device device;
  metal::Submission submission(device);
  const Delay delay(device, submission);
  metal::Renderer renderer(device, submission, tests::fixture_denoised_graph(),
                           &scene);
  std::vector<std::unique_ptr<metal::Offscreen>> targets;
  targets.reserve(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    targets.push_back(
        std::make_unique<metal::Offscreen>(device, submission, size));
  }
  const std::uint64_t first = submission.next_sequence();
  HandOffRun run;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto pool = metal::scoped_pool();
    const frame::FrameInputs& inputs = frames[i];
    renderer.prepare(inputs, size);
    const metal::FrameSlot begun = submission.begin(parts);
    REQUIRE((begun.second != nullptr) == (parts == metal::Parts::two));
    if (begun.second != nullptr) {
      delay.record(begun.second);
    }
    renderer.record(begun, inputs, targets[i]->texture(), size);
    submission.commit();
    if (begun.settled && begun.settled->sequence >= first) {
      run.records.push_back(span(*begun.settled));
    }
    if (parts == metal::Parts::one) {
      if (const auto settled = submission.wait_until_complete(begun.sequence)) {
        run.records.push_back(span(*settled));
      }
    }
  }
  for (const metal::Completed& settled : submission.drain()) {
    run.records.push_back(span(settled));
  }
  std::ranges::sort(run.records, {}, &FrameSpan::sequence);
  for (const std::unique_ptr<metal::Offscreen>& target : targets) {
    run.shown.push_back(tests::read_back(*target));
  }
  run.last.assign(std::size_t{size.width} * size.height * 4, 0.0f);
  renderer.read_accumulated(run.last);
  return run;
}

}  // namespace serenity::tests
