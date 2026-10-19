#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace serenity::cuda {

// Axis: GPU backend.
//
// The submission's bookkeeping (submission.h), apart from the driver: which
// sequence and slot a frame takes, which committed frame each slot still
// holds unsettled, which frames are settled, and the misuses refused. No
// driver call, so its rules are a unit, tested as one (A.1: what changes
// with the driver apart from what does not; F.8). The submission does the
// waiting, on the events, and tells it what it settled.
//
// A frame's sequence counts from 0 and its slot is sequence %
// frames_in_flight. Frames complete in order on the submission's one
// stream, so settling the oldest unsettled frame first settles them all in
// order, and a frame is settled once it, or a later one, has been waited
// for.
//
// Throws CudaError for a misuse: a frame begun while one is open, a commit
// with none open, a wait for a frame not committed.
inline constexpr std::uint32_t frames_in_flight = 2;

class FrameSequence {
 public:
  // A frame begun: its slot and sequence, and the earlier frame that slot
  // still holds unsettled, which the submission settles before it is
  // written again (GPU.7).
  struct Begun {
    std::uint32_t slot = 0;
    std::uint64_t sequence = 0;
    std::optional<std::uint64_t> settle_first;
  };

  Begun begin();
  // The slot of the frame committed, the one begun last.
  std::uint32_t commit();

  // Refuses a wait for frame `sequence` if it has not been committed.
  void check_committed(std::uint64_t sequence) const;

  // The slot of the oldest committed frame up to `sequence` not yet
  // settled; none once every one is.
  std::optional<std::uint32_t> oldest_unsettled(std::uint64_t sequence) const;

  // `slot`'s frame is settled, and so is every frame before it.
  void settled(std::uint32_t slot);

  // Whether frame `sequence` is known complete; and the frame `slot` holds
  // unsettled, if any.
  bool is_settled(std::uint64_t sequence) const noexcept {
    return sequence < settled_through_;
  }
  std::optional<std::uint64_t> pending(std::uint32_t slot) const {
    return pending_.at(slot);
  }

  // The sequence the next begin() hands out.
  std::uint64_t next() const noexcept { return next_; }

 private:
  std::array<std::optional<std::uint64_t>, frames_in_flight> pending_{};
  std::uint64_t next_ = 0;
  std::uint64_t settled_through_ = 0;  // every frame before it is settled
  bool open_ = false;                  // a frame begun and not committed
};

}  // namespace serenity::cuda
