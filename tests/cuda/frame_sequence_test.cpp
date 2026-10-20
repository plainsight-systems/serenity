// The submission's bookkeeping (cuda/device/frame_sequence.h), a unit with
// no driver: sequences from 0 and slots by sequence, the frame a slot still
// holds settled before it is written again, waits settling oldest first, a
// frame's parts and its hand-off, and the misuses refused.

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
    const FrameSequence::Begun begun = frames.begin(Parts::one);
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
    CHECK(frames.commit().slot == begun.slot);
  }
  CHECK(frames.next() == 5);
}

TEST_CASE("a wait settles every frame up to it, oldest first") {
  FrameSequence frames;
  for (int n = 0; n < 2; ++n) {
    (void)frames.begin(Parts::one);
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
    (void)frames.begin(Parts::one);
    (void)frames.commit();
  }
  frames.settled(1);  // frame 1 completed, so frame 0 did: frames end in order
  CHECK(frames.is_settled(0));
  CHECK_FALSE(frames.pending(0));
  CHECK_FALSE(frames.oldest_unsettled(1));
}

TEST_CASE("the bookkeeping refuses what the protocol does not allow") {
  FrameSequence frames;
  CHECK_THROWS_AS((void)frames.commit(), CudaError);  // nothing begun
  (void)frames.begin(Parts::one);
  CHECK_THROWS_AS((void)frames.begin(Parts::one), CudaError);  // begun twice
  CHECK_THROWS_AS(frames.check_committed(0), CudaError);       // not committed
  (void)frames.commit();
  CHECK_NOTHROW(frames.check_committed(0));
  CHECK_THROWS_AS(frames.check_committed(1), CudaError);
}

TEST_CASE("a frame of two parts is handed off once, between its halves") {
  FrameSequence frames;
  const FrameSequence::Begun begun = frames.begin(Parts::two);
  CHECK_THROWS_AS((void)frames.commit(), CudaError);  // not handed off
  CHECK(frames.hand_off() == begun.slot);
  CHECK_THROWS_AS((void)frames.hand_off(), CudaError);  // handed off twice
  const FrameSequence::Committed committed = frames.commit();
  CHECK(committed.slot == begun.slot);
  CHECK(committed.parts == Parts::two);
  CHECK_THROWS_AS((void)frames.hand_off(), CudaError);  // nothing begun
}

TEST_CASE("a frame of one part is never handed off") {
  FrameSequence frames;
  (void)frames.begin(Parts::one);
  CHECK_THROWS_AS((void)frames.hand_off(), CudaError);
  CHECK(frames.commit().parts == Parts::one);
}

TEST_CASE(
    "a frame of one part after one of two starts after that frame's image "
    "half; no other frame waits for one") {
  FrameSequence frames;
  // one after one, two after one, two after two: each half follows its own
  // half of the frame before on its stream, so none waits.
  CHECK_FALSE(frames.begin(Parts::one).after_image_half);
  (void)frames.commit();
  CHECK_FALSE(frames.begin(Parts::two).after_image_half);
  (void)frames.hand_off();
  (void)frames.commit();
  const FrameSequence::Begun two_after_two = frames.begin(Parts::two);
  CHECK_FALSE(two_after_two.after_image_half);
  (void)frames.hand_off();
  (void)frames.commit();
  // one after two: the light stream waits for frame 2's image half.
  const FrameSequence::Begun one_after_two = frames.begin(Parts::one);
  CHECK(one_after_two.sequence == 3);
  CHECK(one_after_two.after_image_half ==
        std::optional<std::uint32_t>{two_after_two.slot});
  (void)frames.commit();
  CHECK_FALSE(frames.begin(Parts::one).after_image_half);
}

}  // namespace
}  // namespace serenity::cuda
