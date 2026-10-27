#include "metal/device/frame_sequence.h"

#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

FrameSequence::Begun FrameSequence::to_begin() const {
  if (finished_) {
    throw MetalError("begin() after finish()");
  }
  if (open_) {
    throw MetalError("begin() while submission " + std::to_string(next_ - 1) +
                     " is still open");
  }
  const std::uint64_t sequence = next_;
  Begun begun{static_cast<std::uint32_t>(sequence % frames_in_flight), sequence,
              std::nullopt};
  if (sequence >= frames_in_flight) {
    begun.settle_first = sequence - frames_in_flight;
  }
  return begun;
}

void FrameSequence::opened() noexcept {
  open_ = true;
  ++next_;
}

std::uint64_t FrameSequence::to_hand_off() const {
  if (!open_) {
    throw MetalError("a hand-off with no submission begun");
  }
  return next_ - 1;
}

std::uint64_t FrameSequence::commit() {
  if (!open_) {
    throw MetalError("commit with no submission begun");
  }
  if (abandoned_) {
    throw MetalError("submission " + std::to_string(next_ - 1) +
                     " cannot be committed: memory it may use was released "
                     "while it was open");
  }
  open_ = false;
  return next_ - 1;
}

void FrameSequence::check_waitable(std::uint64_t sequence) const {
  if (sequence >= next_ || (open_ && sequence == next_ - 1)) {
    throw MetalError("wait_until_complete(" + std::to_string(sequence) +
                     ") for a submission not committed");
  }
  if (sequence + frames_in_flight < next_) {
    throw MetalError("wait_until_complete(" + std::to_string(sequence) +
                     "): its slot has been reused");
  }
}

std::uint64_t FrameSequence::drain_from() const {
  if (open_) {
    throw MetalError("drain() or finish() while submission " +
                     std::to_string(next_ - 1) + " is still open");
  }
  // Every one before them was settled when its slot was reused.
  return next_ > frames_in_flight ? next_ - frames_in_flight : 0;
}

bool FrameSequence::is_settled(std::uint64_t sequence) const {
  return settled_through_.at(sequence % frames_in_flight) >= sequence + 1;
}

void FrameSequence::settled(std::uint64_t sequence) {
  settled_through_.at(sequence % frames_in_flight) = sequence + 1;
}

void FrameSequence::released() noexcept {
  if (open_) {
    abandoned_ = true;  // it may have recorded a use of what was released
  }
}

}  // namespace serenity::metal
