#pragma once

#include <cuda.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Film (what a pixel accumulates: its failures).
//
// A count of one kind of failure a frame's programs report on the GPU,
// Metal's FailureCount (metal/film/failure_count.h, which says what each
// counts and why each is counted apart) on a GPU whose memory is not the
// CPU's. The renderer keeps one today, samples a converging pass left out
// for not being finite (cuda/film/accumulate.cuh); MNEE's lost requests
// join it with MNEE (Phase 4).
//
// One 32-bit counter a frame slot on the device, which the frame's programs
// add to, and one in pinned host memory, which the frame's own work fills:
//
//   - begin_frame(frame), as a frame is recorded into a slot, first adds the
//     slot's host count to a 64-bit total kept on the CPU (the slot's last
//     frame has completed: Submission::begin() waited for it, and its copy
//     back with it), then zeroes the slot's device counter on the frame's
//     stream and remembers which frame the slot now counts for;
//   - end_frame(frame), after the frame's last pass and before its commit,
//     copies the slot's device counter to its host count on the stream, so
//     the copy completes with the frame;
//   - count() is the total, plus each slot's host count whose frame has
//     completed (Submission::has_completed, which does not wait), so it is
//     exact for every completed frame and never reads a count still on its
//     way.
//
// A frame's counter holds at most its pixels' samples, under 2^32 at any
// size the backend renders; the total is 64-bit. Throws CudaError if the
// memory cannot be made, or for a slot that is not one. Not
// performance-sensitive: a few bytes, twice a frame.
class FailureCount {
 public:
  // `what` names what it counts, for its errors: "samples not finite".
  FailureCount(Submission& submission, std::string_view what);

  FailureCount(const FailureCount&) = delete;
  FailureCount& operator=(const FailureCount&) = delete;
  FailureCount(FailureCount&&) = delete;
  FailureCount& operator=(FailureCount&&) = delete;
  ~FailureCount() = default;

  // Before and after frame `frame`'s passes; see above.
  void begin_frame(const FrameSlot& frame);
  void end_frame(const FrameSlot& frame);

  // Slot `slot`'s device counter, as a program adds to it.
  CUdeviceptr address(std::uint32_t slot) const;

  // The failures over every completed frame; see above.
  std::uint64_t count() const noexcept;

 private:
  std::uint32_t host_count(std::uint32_t slot) const noexcept;

  const Submission& submission_;
  std::string what_;
  DeviceBuffer counters_;  // one uint32 a slot
  PinnedBuffer copies_;    // one uint32 a slot
  // sequence + 1 each slot counts for; 0 for none
  std::array<std::uint64_t, frames_in_flight> counting_{};
  std::uint64_t total_ = 0;
};

}  // namespace serenity::cuda
