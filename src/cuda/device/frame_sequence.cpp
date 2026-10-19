#include "cuda/device/frame_sequence.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

FrameSequence::Begun FrameSequence::begin() {
  if (open_) {
    throw CudaError("Submission::begin: a frame is begun and not committed");
  }
  const auto slot = static_cast<std::uint32_t>(next_ % frames_in_flight);
  open_ = true;
  return Begun{slot, next_, pending_.at(slot)};
}

std::uint32_t FrameSequence::commit() {
  if (!open_) {
    throw CudaError("Submission::commit: no frame is begun");
  }
  const auto slot = static_cast<std::uint32_t>(next_ % frames_in_flight);
  pending_.at(slot) = next_;
  ++next_;
  open_ = false;
  return slot;
}

void FrameSequence::check_committed(std::uint64_t sequence) const {
  if (sequence >= next_) {
    throw CudaError("Submission::wait_until_complete: frame " +
                    std::to_string(sequence) + " is not committed");
  }
}

std::optional<std::uint32_t> FrameSequence::oldest_unsettled(
    std::uint64_t sequence) const {
  std::optional<std::uint32_t> oldest;
  std::optional<std::uint64_t> oldest_sequence;
  for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    const std::optional<std::uint64_t> held = pending_.at(slot);
    if (held && *held <= sequence &&
        (!oldest_sequence || *held < *oldest_sequence)) {
      oldest = slot;
      oldest_sequence = held;
    }
  }
  return oldest;
}

void FrameSequence::settled(std::uint32_t slot) {
  const std::optional<std::uint64_t> held = pending_.at(slot);
  if (!held) {
    return;
  }
  // Every frame up to it completed before it, on the one stream.
  for (std::optional<std::uint64_t>& other : pending_) {
    if (other && *other <= *held) {
      other.reset();
    }
  }
  if (*held + 1 > settled_through_) {
    settled_through_ = *held + 1;
  }
}

}  // namespace serenity::cuda
