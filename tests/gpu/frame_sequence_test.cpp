// The submission's bookkeeping (metal/device/frame_sequence.h), a unit with
// no Metal call: sequences and slots, the submission a slot held settled
// before reuse, settling once, what an idle wait waits for, and the misuses
// Metal's protocol refuses, each with its words.

#include "metal/device/frame_sequence.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <string>

#include "metal/device/error.h"
#include "support/text.h"

namespace serenity::metal {
namespace {

// Begins and commits one submission; its sequence.
std::uint64_t submit(FrameSequence& frames) {
  const FrameSequence::Begun begun = frames.to_begin();
  if (begun.settle_first) {
    frames.settled(*begun.settle_first);  // as the submission settles it
  }
  frames.opened();
  return frames.commit();
}

TEST_CASE("submissions take sequences in order, slots by sequence") {
  FrameSequence frames;
  for (std::uint64_t n = 0; n < 5; ++n) {
    const FrameSequence::Begun begun = frames.to_begin();
    CHECK(begun.sequence == n);
    CHECK(begun.slot == n % frames_in_flight);
    // The slot's last submission, two back, settled before reuse.
    if (n >= frames_in_flight) {
      CHECK(begun.settle_first == n - frames_in_flight);
      frames.settled(n - frames_in_flight);
    } else {
      CHECK_FALSE(begun.settle_first);
    }
    frames.opened();
    CHECK(frames.commit() == n);
  }
  CHECK(frames.next() == 5);
}

TEST_CASE("each submission is settled once") {
  FrameSequence frames;
  (void)submit(frames);
  CHECK_FALSE(frames.is_settled(0));
  frames.settled(0);
  CHECK(frames.is_settled(0));
  (void)submit(frames);
  (void)submit(frames);  // settles 0 again: still settled, once
  CHECK(frames.is_settled(0));
  CHECK_FALSE(frames.is_settled(1));
  CHECK_FALSE(frames.is_settled(2));
}

TEST_CASE("an idle wait waits for what is committed, never what is open") {
  FrameSequence frames;
  CHECK(frames.committed() == 0);
  (void)submit(frames);
  CHECK(frames.committed() == 1);
  (void)frames.to_begin();
  frames.opened();
  CHECK(frames.committed() == 1);  // the open one is not waited for
  (void)frames.commit();
  CHECK(frames.committed() == 2);
}

TEST_CASE("a drain settles the last frames_in_flight committed") {
  FrameSequence frames;
  CHECK(frames.drain_from() == 0);
  for (int n = 0; n < 5; ++n) {
    (void)submit(frames);
  }
  CHECK(frames.drain_from() == 5 - frames_in_flight);
}

TEST_CASE("the protocol refuses misuse, each in its words") {
  FrameSequence frames;
  CHECK_THROWS_WITH_AS((void)frames.commit(), "commit with no submission begun",
                       MetalError);
  (void)frames.to_begin();
  frames.opened();
  CHECK_THROWS_WITH_AS((void)frames.to_begin(),
                       "begin() while submission 0 is still open", MetalError);
  CHECK_THROWS_AS(frames.check_waitable(0), MetalError);   // not committed
  CHECK_THROWS_AS((void)frames.drain_from(), MetalError);  // one open
  (void)frames.commit();
  CHECK_NOTHROW(frames.check_waitable(0));
  CHECK_THROWS_AS(frames.check_waitable(5), MetalError);
  for (int n = 0; n < 3; ++n) {
    (void)submit(frames);
  }
  CHECK_THROWS_WITH_AS(frames.check_waitable(0),
                       "wait_until_complete(0): its slot has been reused",
                       MetalError);
  frames.finish();
  CHECK_THROWS_WITH_AS((void)frames.to_begin(), "begin() after finish()",
                       MetalError);
}

TEST_CASE("a hand-off names the submission open, and none with none open") {
  FrameSequence frames;
  CHECK_THROWS_WITH_AS((void)frames.to_hand_off(),
                       "a hand-off with no submission begun", MetalError);
  (void)submit(frames);
  (void)frames.to_begin();
  frames.opened();
  CHECK(frames.to_hand_off() == 1);
  (void)frames.commit();
  CHECK_THROWS_AS((void)frames.to_hand_off(), MetalError);
}

TEST_CASE("a submission open when memory is released is never committed") {
  FrameSequence frames;
  frames.released();  // none open: nothing abandoned
  (void)submit(frames);
  (void)frames.to_begin();
  frames.opened();
  frames.released();
  try {
    (void)frames.commit();
    FAIL("an abandoned submission was committed");
  } catch (const MetalError& refused) {
    CHECK(tests::contains(refused.what(), "cannot be committed"));
  }
}

}  // namespace
}  // namespace serenity::metal
