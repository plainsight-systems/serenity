#include "cuda/device/submission.h"

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
  if (!pending_.at(slot)) {
    return;
  }
  const std::uint64_t sequence = *pending_.at(slot);
  pending_.at(slot).reset();
  check(cuEventSynchronize(events_.at(slot)),
        "frame " + std::to_string(sequence) + " on the GPU");
  if (sequence + 1 > settled_through_) {
    settled_through_ = sequence + 1;
  }
}

FrameSlot Submission::begin() {
  if (open_) {
    throw CudaError("Submission::begin: a frame is begun and not committed");
  }
  const auto slot = static_cast<std::uint32_t>(next_ % frames_in_flight);
  settle(slot);
  open_ = true;
  return FrameSlot{stream_, slot, next_};
}

void Submission::commit() {
  if (!open_) {
    throw CudaError("Submission::commit: no frame is begun");
  }
  const auto slot = static_cast<std::uint32_t>(next_ % frames_in_flight);
  check(cuEventRecord(events_.at(slot), stream_), "cuEventRecord");
  pending_.at(slot) = next_;
  ++next_;
  open_ = false;
}

void Submission::wait_until_complete(std::uint64_t sequence) {
  if (sequence >= next_) {
    throw CudaError("Submission::wait_until_complete: frame " +
                    std::to_string(sequence) + " is not committed");
  }
  // Frames complete in order on the one stream: settling every pending one
  // up to `sequence`, oldest first, settles it.
  for (std::uint64_t s = settled_through_; s <= sequence; ++s) {
    const auto slot = static_cast<std::uint32_t>(s % frames_in_flight);
    if (pending_.at(slot) == s) {
      settle(slot);
    }
  }
  if (settled_through_ <= sequence) {
    settled_through_ = sequence + 1;
  }
}

void Submission::finish() {
  if (next_ > 0) {
    wait_until_complete(next_ - 1);
  }
}

bool Submission::has_completed(std::uint64_t sequence) const noexcept {
  if (sequence >= next_) {
    return false;
  }
  if (sequence < settled_through_) {
    return true;
  }
  const auto slot = static_cast<std::uint32_t>(sequence % frames_in_flight);
  return pending_.at(slot) == sequence &&
         cuEventQuery(events_.at(slot)) == CUDA_SUCCESS;
}

void Submission::wait_idle() noexcept {
  if (stream_ != nullptr) {
    (void)cuStreamSynchronize(stream_);
  }
}

}  // namespace serenity::cuda
