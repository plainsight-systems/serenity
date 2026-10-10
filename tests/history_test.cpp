// The accumulated image's history (core/frame/history.h): which frames may
// join it, and the window's and the headless renderer's plans for when it
// starts over. No GPU: the rule is the core's.

#include <limits>
#include <stdexcept>
#include <string>

#include <doctest/doctest.h>

#include "core/frame/history.h"

using namespace serenity;
using frame::Extent;
using frame::FrameInputs;
using frame::History;
using frame::HistoryError;
using frame::Seconds;

namespace {

constexpr Extent small{16, 16};
constexpr Extent large{24, 16};

FrameInputs frame_at(std::uint64_t index, std::uint64_t since, double time = 0.0) {
    return FrameInputs{.time = Seconds(time), .index = index, .accumulated_since = since, .camera = std::nullopt};
}

}  // namespace

TEST_CASE("an image holds the frames since it started over, at one size") {
    History history;
    const frame::Joined first = history.join(frame_at(0, 0), small, false);
    CHECK(first.held == 0);
    CHECK(first.remade);
    CHECK(history.join(frame_at(1, 0), small, false).held == 1);
    const frame::Joined third = history.join(frame_at(2, 0), small, false);
    CHECK(third.held == 2);
    CHECK_FALSE(third.remade);

    // Refused, and nothing changes: the next frame still joins.
    CHECK_THROWS_AS(history.join(frame_at(4, 0), small, false), HistoryError);  // a frame skipped
    CHECK_THROWS_AS(history.join(frame_at(3, 0), large, false), HistoryError);  // a size changed
    CHECK_THROWS_AS(history.join(frame_at(3, 1), small, false), HistoryError);  // since moved
    CHECK_THROWS_AS(history.join(frame_at(3, 5), small, false), HistoryError);  // since after the frame
    CHECK(history.join(frame_at(3, 0), small, false).held == 3);

    // Starting over at the same size keeps the image; at another, remakes it.
    CHECK_FALSE(history.join(frame_at(4, 4), small, false).remade);
    const frame::Joined resized = history.join(frame_at(5, 5), large, false);
    CHECK(resized.held == 0);
    CHECK(resized.remade);
}

TEST_CASE("a moving scene's image holds one instant; a still scene's, any") {
    History moving;
    (void)moving.join(frame_at(0, 0, 1.5), small, true);
    CHECK(moving.join(frame_at(1, 0, 1.5), small, true).held == 1);
    CHECK_THROWS_WITH_AS(moving.join(frame_at(2, 0, 1.6), small, true), doctest::Contains("one instant"),
                         HistoryError);
    CHECK(moving.join(frame_at(2, 2, 1.6), small, true).held == 0);

    History still;
    (void)still.join(frame_at(0, 0, 1.5), small, false);
    CHECK(still.join(frame_at(1, 0, 9.0), small, false).held == 1);
}

TEST_CASE("an image holds at most 2^24 - 1 frames") {
    History history;
    CHECK_THROWS_WITH_AS(history.join(frame_at(frame::max_accumulated_frames + 1, 0), small, false),
                         doctest::Contains("at most"), HistoryError);
}

TEST_CASE("the window starts over when its view changes, every frame while the scene moves, and when full") {
    frame::LiveHistory still(false);
    CHECK(still.since(0, false) == 0);
    CHECK(still.since(1, false) == 0);
    CHECK(still.since(2, true) == 2);   // resized
    CHECK(still.since(3, false) == 2);
    CHECK(still.since(2 + frame::max_accumulated_frames, false) == 2);
    CHECK(still.since(3 + frame::max_accumulated_frames, false) == 3 + frame::max_accumulated_frames);  // full

    frame::LiveHistory moving(true);
    for (std::uint64_t i = 0; i < 5; ++i) {
        CHECK(moving.since(i, false) == i);
    }
}

TEST_CASE("headless: samples of one instant, the run converging only while the scene looks the same") {
    // A still scene: frames 3 and 4, four samples each, one image from frame
    // 3's first sample on.
    const frame::HeadlessPlan still = frame::plan_headless(3, 4, true, false, false);
    CHECK(still.samples == 4);
    CHECK(still.sample(3, 0).index == 12);
    CHECK(still.sample(3, 3).index == 15);
    CHECK(still.sample(4, 0).index == 16);
    CHECK(still.sample(4, 2).accumulated_since == 12);

    // A moving scene, time advancing: each frame its own image.
    const frame::HeadlessPlan moving = frame::plan_headless(3, 4, true, true, false);
    CHECK(moving.sample(4, 0).accumulated_since == 16);
    CHECK(moving.sample(4, 3).accumulated_since == 16);
    CHECK(moving.sample(4, 3).index == 19);

    // Time frozen: the moving scene looks the same at every frame.
    CHECK(frame::plan_headless(3, 4, true, true, true).sample(4, 0).accumulated_since == 12);

    // A graph that accumulates nothing: one sample a frame, whatever was asked.
    const frame::HeadlessPlan once = frame::plan_headless(3, 64, false, true, false);
    CHECK(once.samples == 1);
    CHECK(once.sample(4, 0).index == 4);
    CHECK(once.sample(4, 0).accumulated_since == 4);
}

TEST_CASE("the plans refuse what would wrap an index: frames out of order, a sample past the frame's, too many") {
    frame::LiveHistory live(false);
    CHECK(live.since(10, true) == 10);
    CHECK_THROWS_AS(live.since(9, false), HistoryError);  // before the image's first: index - since would wrap

    const frame::HeadlessPlan plan = frame::plan_headless(0, 4, true, false, false);
    CHECK_THROWS_AS(plan.sample(3, 4), std::invalid_argument);
    CHECK(plan.sample(3, 3).index == 15);
    const std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(plan.sample(most / 4, 0), std::overflow_error);
    CHECK(plan.sample(most / 4 - 1, 3).index == most / 4 * 4 - 1);
    const frame::HeadlessPlan none = frame::plan_headless(0, 0, true, false, false);
    CHECK_THROWS_AS(none.sample(0, 0), std::invalid_argument);
}
