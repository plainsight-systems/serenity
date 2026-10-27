// A submission's hand-off (metal/device/submission.h, "A hand-off"): work
// off the GPU between two halves of a submission's last part, on either
// queue. The worker sees what the part before it wrote, the part after sees
// what the worker wrote, a worker's failure is the submission's, and the
// misuses are refused. An integration test of the queue's order: a worker
// that sleeps before it writes would be overtaken by a part after that did
// not wait for it.

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/submission.h"
#include "support/text.h"

namespace metal = serenity::metal;
namespace tests = serenity::tests;

namespace {

// Long beside a copy of four bytes: a part after that did not wait for the
// worker would have run in it.
constexpr std::chrono::milliseconds worker_sleep{50};

struct Rig {
  metal::Device device;
  metal::Submission submission{device};
  NS::SharedPtr<MTL::SharedEventListener> listener =
      NS::TransferPtr(MTL::SharedEventListener::alloc()->init());

  // A word of shared memory, resident, holding `value`.
  NS::SharedPtr<MTL::Buffer> word(std::uint32_t value) {
    auto buffer = NS::TransferPtr(device.handle()->newBuffer(
        sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    REQUIRE(buffer);
    *static_cast<std::uint32_t*>(buffer->contents()) = value;
    submission.make_resident(buffer.get());
    return buffer;
  }
};

std::uint32_t read(MTL::Buffer* buffer) {
  return *static_cast<const std::uint32_t*>(buffer->contents());
}

void copy(MTL4::CommandBuffer* commands, const MTL::Buffer* from,
          const MTL::Buffer* to) {
  MTL4::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
  REQUIRE(encoder != nullptr);
  encoder->copyFromBuffer(from, 0, to, 0, sizeof(std::uint32_t));
  encoder->endEncoding();
}

}  // namespace

TEST_CASE(
    "work after a hand-off sees what its worker wrote, and the worker what "
    "the part before wrote") {
  for (const metal::Parts parts : {metal::Parts::one, metal::Parts::two}) {
    INFO((parts == metal::Parts::one ? "one part" : "two parts"));
    Rig rig;
    const auto a = rig.word(7);
    const auto b = rig.word(0);
    const auto c = rig.word(0);
    const auto d = rig.word(0);
    const metal::FrameSlot begun = rig.submission.begin(parts);
    MTL4::CommandBuffer* last =
        parts == metal::Parts::two ? begun.second : begun.commands;
    copy(last, a.get(), b.get());  // the part before: b = 7
    const metal::HandOff hand_off = rig.submission.hand_off();
    std::atomic<std::uint32_t> seen{0};
    hand_off.ready->notifyListener(
        rig.listener.get(), hand_off.value,
        [&seen, b = b.get(), c = c.get(), report = hand_off.report,
         done = hand_off.done,
         value = hand_off.value](MTL::SharedEvent*, std::uint64_t) {
          seen = read(b);
          std::this_thread::sleep_for(worker_sleep);
          *static_cast<std::uint32_t*>(c->contents()) = 9;
          report->succeeded(value - 1);
          done->setSignaledValue(value);
        });
    copy(hand_off.after, c.get(), d.get());  // the part after: d = c
    rig.submission.commit();
    const auto record = rig.submission.wait_until_complete(begun.sequence);
    CHECK(seen == 7);
    CHECK(read(d.get()) == 9);
    // The record spans the worker: from the part before's start to the part
    // after's end.
    REQUIRE(record.has_value());
    const metal::Completed spanned = record.value_or(metal::Completed{});
    CHECK(spanned.gpu_end - spanned.gpu_start >=
          std::chrono::duration<double>(worker_sleep));
    (void)rig.submission.finish();
  }
}

TEST_CASE("a worker's failure is its submission's, in its words") {
  Rig rig;
  const metal::FrameSlot begun = rig.submission.begin();
  const metal::HandOff hand_off = rig.submission.hand_off();
  hand_off.ready->notifyListener(
      rig.listener.get(), hand_off.value,
      [report = hand_off.report, done = hand_off.done, value = hand_off.value](
          MTL::SharedEvent*, std::uint64_t) {
        report->failed(value - 1, "the worker's own words");
        done->setSignaledValue(value);
      });
  rig.submission.commit();
  const std::string error = tests::error_of<metal::MetalError>(
      [&] { return rig.submission.wait_until_complete(begun.sequence); });
  CHECK(error == "submission 0's hand-off failed: the worker's own words");
}

TEST_CASE("a hand-off with none open, or a second, is refused") {
  Rig rig;
  CHECK_THROWS_WITH_AS((void)rig.submission.hand_off(),
                       "a hand-off with no submission begun",
                       metal::MetalError);
  const metal::FrameSlot begun = rig.submission.begin();
  const metal::HandOff hand_off = rig.submission.hand_off();
  CHECK_THROWS_WITH_AS((void)rig.submission.hand_off(),
                       "hand_off(): submission 0 has been handed off already",
                       metal::MetalError);
  // The first goes on: its worker done at once.
  hand_off.report->succeeded(0);
  hand_off.done->setSignaledValue(hand_off.value);
  rig.submission.commit();
  CHECK_NOTHROW((void)rig.submission.wait_until_complete(begun.sequence));
  (void)rig.submission.finish();
}
