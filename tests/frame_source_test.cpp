// The rule every frame source applies to the rate the window asks for
// (core/presentation/frame_source.h, frame_rate_refusal): finite, positive,
// and minimum <= preferred <= maximum.

#include "core/presentation/frame_source.h"

#include <doctest/doctest.h>

#include <limits>

namespace serenity::presentation {
namespace {

TEST_CASE("a frame rate range is refused unless positive and in order") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const FrameRate wrong :
       {FrameRate{0.0F, 0.0F, 0.0F}, FrameRate{-60.0F, 60.0F, 60.0F},
        FrameRate{60.0F, 30.0F, 45.0F}, FrameRate{30.0F, 60.0F, 90.0F},
        FrameRate{30.0F, 60.0F, 20.0F}, FrameRate{30.0F, 60.0F, nan},
        FrameRate{30.0F, infinity, 60.0F}}) {
    INFO("rate " << wrong.minimum << " to " << wrong.maximum << ", preferred "
                 << wrong.preferred);
    CHECK(frame_rate_refusal(wrong));
  }
  CHECK_FALSE(frame_rate_refusal(FrameRate{60.0F, 60.0F, 60.0F}));
  CHECK_FALSE(frame_rate_refusal(FrameRate{30.0F, 60.0F, 60.0F}));
}

}  // namespace
}  // namespace serenity::presentation
