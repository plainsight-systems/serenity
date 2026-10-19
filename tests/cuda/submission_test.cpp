// The CUDA submission's protocol (cuda/device/submission.h): sequences from
// 0, slots by sequence, a frame known complete once waited for, and the
// misuses it refuses rather than lets through.

#include "cuda/device/submission.h"

#include <doctest/doctest.h>

#include <cstdint>

#include "cuda/device/device.h"
#include "cuda/device/error.h"

namespace serenity::cuda {
namespace {

TEST_CASE("frames take sequences in order and slots by sequence") {
  const Device device;
  Submission submission(device);
  for (std::uint64_t n = 0; n < 5; ++n) {
    const FrameSlot begun = submission.begin();
    CHECK(begun.sequence == n);
    CHECK(begun.slot == n % frames_in_flight);
    CHECK(begun.stream != nullptr);
    submission.commit();
  }
  CHECK(submission.next_sequence() == 5);
  submission.wait_until_complete(4);
  for (std::uint64_t n = 0; n < 5; ++n) {
    CHECK(submission.has_completed(n));
  }
  CHECK_FALSE(submission.has_completed(5));  // never committed
}

TEST_CASE("a submission refuses what its protocol does not allow") {
  const Device device;
  Submission submission(device);
  CHECK_THROWS_AS(submission.commit(), CudaError);  // nothing begun
  (void)submission.begin();
  CHECK_THROWS_AS((void)submission.begin(), CudaError);  // begun twice
  submission.commit();
  CHECK_THROWS_AS(submission.wait_until_complete(1), CudaError);  // not yet
  CHECK_NOTHROW(submission.finish());
  CHECK(submission.has_completed(0));
}

}  // namespace
}  // namespace serenity::cuda
