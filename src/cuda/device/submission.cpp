#include "cuda/device/submission.h"

#include <optional>
#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

Submission::Submission(const Device& /*device*/) {
  // The device's primary context is current on this thread (device.h).
  check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  for (CUevent& event : events_) {
    if (const CUresult made = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
        made != CUDA_SUCCESS) {
      wait_idle();
      for (CUevent& made_one : events_) {
        if (made_one != nullptr) {
          (void)cuEventDestroy(made_one);
        }
      }
      (void)cuStreamDestroy(stream_);
      check(made, "cuEventCreate");
    }
  }
}

Submission::~Submission() {
  wait_idle();
  for (CUevent event : events_) {
    (void)cuEventDestroy(event);
  }
  (void)cuStreamDestroy(stream_);
}

void Submission::settle(std::uint32_t slot) {
  const std::optional<std::uint64_t> pending = frames_.pending(slot);
  if (!pending) {
    return;
  }
  check(cuEventSynchronize(events_.at(slot)),
        "frame " + std::to_string(*pending) + " on the GPU");
  frames_.settled(slot);
}

FrameSlot Submission::begin() {
  const FrameSequence::Begun begun = frames_.begin();
  if (begun.settle_first) {
    settle(begun.slot);
  }
  return FrameSlot{stream_, begun.slot, begun.sequence};
}

void Submission::commit() {
  const auto slot =
      static_cast<std::uint32_t>(frames_.next() % frames_in_flight);
  check(cuEventRecord(events_.at(slot), stream_), "cuEventRecord");
  (void)frames_.commit();
}

void Submission::wait_until_complete(std::uint64_t sequence) {
  frames_.check_committed(sequence);
  // Oldest first: frames complete in order on the one stream.
  while (const std::optional<std::uint32_t> slot =
             frames_.oldest_unsettled(sequence)) {
    settle(*slot);
  }
}

void Submission::finish() {
  if (frames_.next() > 0) {
    wait_until_complete(frames_.next() - 1);
  }
}

bool Submission::has_completed(std::uint64_t sequence) const noexcept {
  if (sequence >= frames_.next()) {
    return false;
  }
  if (frames_.is_settled(sequence)) {
    return true;
  }
  const auto slot = static_cast<std::uint32_t>(sequence % frames_in_flight);
  return frames_.pending(slot) == sequence &&
         cuEventQuery(events_.at(slot)) == CUDA_SUCCESS;
}

void Submission::wait_idle() noexcept {
  if (stream_ != nullptr) {
    (void)cuStreamSynchronize(stream_);
  }
}

}  // namespace serenity::cuda
