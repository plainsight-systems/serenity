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
// frames_in_flight. Frames complete in order: a frame's last work is after
// the frame before's (below), so settling the oldest unsettled frame first
// settles them all in order, and a frame is settled once it, or a later
// one, has been waited for.
//
// Parts (submission.h, "Two parts"; Metal's metal/device/submission.h). A
// frame is begun in one part or two. A two-part frame is handed off once,
// between its halves, before it is committed; a one-part frame never is.
// A one-part frame begun after a two-part one starts after that frame's
// image half (Begun::after_image_half), which ran on the other stream: the
// two frames share the image half's history (the denoiser's output, the
// radiance image, the bloom; metal/frame/renderer.h, "Two halves"), so the
// light stream waits for it (GPU.8: the wait names that hazard; CP.3).
//
// Throws CudaError for a misuse (I.6, E.2): a frame begun while one is
// open, a hand-off of a one-part frame, a second hand-off, or with no frame
// open, a commit with none open or of a two-part frame not handed off, a
// wait for a frame not committed.
inline constexpr std::uint32_t frames_in_flight = 2;

// How many parts a frame has: one, on the light stream; or two, the second
// on the image stream after the first. A type, not a count or a flag, so
// no other number can be asked for (I.4, Enum.3), a byte as the tidy
// checks ask (performance-enum-size), as Metal's is.
enum class Parts : std::uint8_t { one, two };

class FrameSequence {
 public:
  // A frame begun: its slot and sequence, and the earlier frame that slot
  // still holds unsettled, which the submission settles before it is
  // written again (GPU.7); for a one-part frame after a two-part one, the
  // slot of that frame, whose image half it starts after (above).
  struct Begun {
    std::uint32_t slot = 0;
    std::uint64_t sequence = 0;
    std::optional<std::uint64_t> settle_first;
    std::optional<std::uint32_t> after_image_half;
  };

  Begun begin(Parts parts);
  // The slot of the frame handed off, the one begun last, in two parts.
  std::uint32_t hand_off();
  // The slot of the frame committed, the one begun last, and its parts.
  struct Committed {
    std::uint32_t slot = 0;
    Parts parts = Parts::one;
  };
  Committed commit();

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
  Parts parts_ = Parts::one;           // the open frame's
  bool handed_off_ = false;            // the open frame's hand-off made
  // The last committed frame's parts; one before the first frame.
  Parts last_parts_ = Parts::one;
};

}  // namespace serenity::cuda
