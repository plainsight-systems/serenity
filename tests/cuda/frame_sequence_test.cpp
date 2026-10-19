// The submission's bookkeeping (cuda/device/frame_sequence.h), a unit with
// no driver: sequences from 0 and slots by sequence, the frame a slot still
// holds settled before it is written again, waits settling oldest first, and
// the misuses refused.

#include "cuda/device/frame_sequence.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <optional>

#include "cuda/device/error.h"

namespace serenity::cuda {
namespace {

TEST_CASE("frames take sequences in order and slots by sequence") {
  FrameSequence frames;
  for (std::uint64_t n = 0; n < 5; ++n) {
    const FrameSequence::Begun begun = frames.begin();
    CHECK(begun.sequence == n);
    CHECK(begun.slot == n % frames_in_flight);
    // The slot's last frame, two back, is settled before it is reused, as
    // the submission settles it, before the frame is recorded.
    if (n >= frames_in_flight) {
      CHECK(begun.settle_first == n - frames_in_flight);
      frames.settled(begun.slot);
      CHECK(frames.is_settled(n - frames_in_flight));
    } else {
      CHECK_FALSE(begun.settle_first);
    }
    CHECK(frames.commit() == begun.slot);
  }
  CHECK(frames.next() == 5);
}

TEST_CASE("a wait settles every frame up to it, oldest first") {
  FrameSequence frames;
  for (int n = 0; n < 2; ++n) {
    (void)frames.begin();
    (void)frames.commit();
  }
  CHECK_FALSE(frames.is_settled(0));
  CHECK(frames.oldest_unsettled(1) == std::optional<std::uint32_t>{0});
  frames.settled(0);
  CHECK(frames.is_settled(0));
  CHECK_FALSE(frames.is_settled(1));
  CHECK(frames.oldest_unsettled(1) == std::optional<std::uint32_t>{1});
  frames.settled(1);
  CHECK_FALSE(frames.oldest_unsettled(1));
  CHECK(frames.is_settled(1));
}

TEST_CASE("a later frame settled settles the earlier ones") {
  FrameSequence frames;
  for (int n = 0; n < 2; ++n) {
    (void)frames.begin();
    (void)frames.commit();
  }
  frames.settled(1);  // frame 1 completed, so frame 0 did on the one stream
  CHECK(frames.is_settled(0));
  CHECK_FALSE(frames.pending(0));
  CHECK_FALSE(frames.oldest_unsettled(1));
}

TEST_CASE("the bookkeeping refuses what the protocol does not allow") {
  FrameSequence frames;
  CHECK_THROWS_AS((void)frames.commit(), CudaError);  // nothing begun
  (void)frames.begin();
  CHECK_THROWS_AS((void)frames.begin(), CudaError);       // begun twice
  CHECK_THROWS_AS(frames.check_committed(0), CudaError);  // not committed
  (void)frames.commit();
  CHECK_NOTHROW(frames.check_committed(0));
  CHECK_THROWS_AS(frames.check_committed(1), CudaError);
}

}  // namespace
}  // namespace serenity::cuda
