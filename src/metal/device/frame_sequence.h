#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace serenity::metal {

// Axis: GPU backend.
//
// The submission's bookkeeping (submission.h), apart from Metal: which
// sequence and slot a submission takes, which slots' submissions are
// settled, what has been committed, and the misuses the protocol refuses,
// each with its words. No Metal call, so its rules are a unit, tested as one
// (A.1; F.8), as the CUDA backend's are (cuda/device/frame_sequence.h). The
// submission does the waiting, on its event and its feedback, and tells this
// what it settled.
//
// Metal's rules, which differ from CUDA's: a submission's slot is
// sequence % frames_in_flight, and begin() settles the one two back before
// reusing its slot; a wait for a submission whose slot has been reused is
// refused rather than answered, since its record is gone; nothing begins
// after finish(); and a submission open when memory it may use is released
// is abandoned, never committed (submission.h, Residency).
//
// Throws MetalError (error.h) for a misuse, with the submission's words.
inline constexpr std::uint32_t frames_in_flight = 2;

class FrameSequence {
 public:
  // A submission begun: its slot and sequence, and the earlier submission
  // its slot held, which the submission settles before reusing it.
  struct Begun {
    std::uint32_t slot = 0;
    std::uint64_t sequence = 0;
    std::optional<std::uint64_t> settle_first;
  };

  // The submission the next begin() opens, refused after finish() or while
  // one is open; then, once the submission has settled settle_first and
  // begun its command buffers, opened() opens it. Two steps, so a failure
  // settling leaves nothing begun (E.4).
  Begun to_begin() const;
  void opened() noexcept;
  // The sequence of the submission committed, the one begun last; refused
  // with none open, or for one abandoned.
  std::uint64_t commit();

  // Refused unless submission `sequence` is committed and its slot not
  // reused (wait_until_complete()'s rule).
  void check_waitable(std::uint64_t sequence) const;
  // Refused while a submission is open (drain()'s and finish()'s rule); the
  // submissions that may be unsettled, the last frames_in_flight committed,
  // as [first, next()).
  std::uint64_t drain_from() const;
  // No submission begins after it.
  void finish() noexcept { finished_ = true; }

  // Whether submission `sequence` is settled, and that it now is.
  bool is_settled(std::uint64_t sequence) const noexcept;
  void settled(std::uint64_t sequence) noexcept;

  // Memory a submission may use was released: an open one is abandoned.
  void released() noexcept;

  // The submissions committed: what an idle wait waits for.
  std::uint64_t committed() const noexcept { return open_ ? next_ - 1 : next_; }
  // The sequence the next begin() hands out.
  std::uint64_t next() const noexcept { return next_; }

 private:
  // Each slot's sequence + 1 of its last submission settled.
  std::array<std::uint64_t, frames_in_flight> settled_through_{};
  std::uint64_t next_ = 0;
  bool open_ = false;       // a submission is begun and not yet committed
  bool finished_ = false;   // finish() has run; nothing may be begun after
  bool abandoned_ = false;  // memory was released while open_: never commit
};

}  // namespace serenity::metal
