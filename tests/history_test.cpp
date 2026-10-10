// The accumulated image's history (core/frame/history.h): which frames may
// join it, and the window's and the headless renderer's plans for when it
// starts over. No GPU: the rule is the core's.

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

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
    const frame::Joined first = history.join(frame_at(0, 0), small, frame::SceneMotion::still);
    CHECK(first.held == 0);
    CHECK(first.remade);
    CHECK(history.join(frame_at(1, 0), small, frame::SceneMotion::still).held == 1);
    const frame::Joined third = history.join(frame_at(2, 0), small, frame::SceneMotion::still);
    CHECK(third.held == 2);
    CHECK_FALSE(third.remade);

    // Refused, and nothing changes: the next frame still joins.
    CHECK_THROWS_AS(history.join(frame_at(4, 0), small, frame::SceneMotion::still), HistoryError);  // a frame skipped
    CHECK_THROWS_AS(history.join(frame_at(3, 0), large, frame::SceneMotion::still), HistoryError);  // a size changed
    CHECK_THROWS_AS(history.join(frame_at(3, 1), small, frame::SceneMotion::still), HistoryError);  // since moved
    // since after the frame
    CHECK_THROWS_AS(history.join(frame_at(3, 5), small, frame::SceneMotion::still), HistoryError);
    CHECK(history.join(frame_at(3, 0), small, frame::SceneMotion::still).held == 3);

    // Starting over at the same size keeps the image; at another, remakes it.
    CHECK_FALSE(history.join(frame_at(4, 4), small, frame::SceneMotion::still).remade);
    const frame::Joined resized = history.join(frame_at(5, 5), large, frame::SceneMotion::still);
    CHECK(resized.held == 0);
    CHECK(resized.remade);
}

TEST_CASE("a moving scene's image holds one instant; a still scene's, any") {
    History moving;
    (void)moving.join(frame_at(0, 0, 1.5), small, frame::SceneMotion::changing);
    CHECK(moving.join(frame_at(1, 0, 1.5), small, frame::SceneMotion::changing).held == 1);
    CHECK_THROWS_WITH_AS(moving.join(frame_at(2, 0, 1.6), small, frame::SceneMotion::changing),
                         doctest::Contains("one instant"),
                         HistoryError);
    CHECK(moving.join(frame_at(2, 2, 1.6), small, frame::SceneMotion::changing).held == 0);

    History still;
    (void)still.join(frame_at(0, 0, 1.5), small, frame::SceneMotion::still);
    CHECK(still.join(frame_at(1, 0, 9.0), small, frame::SceneMotion::still).held == 1);
}

TEST_CASE("an image holds at most 2^24 - 1 frames") {
    History history;
    CHECK_THROWS_WITH_AS(history.join(frame_at(frame::max_accumulated_frames + 1, 0), small, frame::SceneMotion::still),
                         doctest::Contains("at most"), HistoryError);
}

TEST_CASE("the window starts over when its view changes, every frame while the scene moves, and when full") {
    frame::LiveHistory still(frame::SceneMotion::still);
    CHECK(still.since(0, frame::View::same) == 0);
    CHECK(still.since(1, frame::View::same) == 0);
    CHECK(still.since(2, frame::View::changed) == 2);   // resized
    CHECK(still.since(3, frame::View::same) == 2);
    CHECK(still.since(2 + frame::max_accumulated_frames, frame::View::same) == 2);
    // Full:
    CHECK(still.since(3 + frame::max_accumulated_frames, frame::View::same) == 3 + frame::max_accumulated_frames);

    frame::LiveHistory moving(frame::SceneMotion::changing);
    for (std::uint64_t i = 0; i < 5; ++i) {
        CHECK(moving.since(i, frame::View::same) == i);
    }
}

TEST_CASE("headless: samples of one instant, the run converging only while the scene looks the same") {
    // A still scene: frames 3 and 4, four samples each, one image from frame
    // 3's first sample on.
    const frame::HeadlessPlan still = frame::plan_headless(
        {.first = 3, .samples = 4, .accumulates = true, .scene_changes = false, .time_frozen = false});
    CHECK(still.samples == 4);
    CHECK(still.sample({.frame = 3, .sample = 0}).index == 12);
    CHECK(still.sample({.frame = 3, .sample = 3}).index == 15);
    CHECK(still.sample({.frame = 4, .sample = 0}).index == 16);
    CHECK(still.sample({.frame = 4, .sample = 2}).accumulated_since == 12);

    // A moving scene, time advancing: each frame its own image.
    const frame::HeadlessPlan moving = frame::plan_headless(
        {.first = 3, .samples = 4, .accumulates = true, .scene_changes = true, .time_frozen = false});
    CHECK(moving.sample({.frame = 4, .sample = 0}).accumulated_since == 16);
    CHECK(moving.sample({.frame = 4, .sample = 3}).accumulated_since == 16);
    CHECK(moving.sample({.frame = 4, .sample = 3}).index == 19);

    // Time frozen: the moving scene looks the same at every frame.
    const frame::HeadlessPlan frozen = frame::plan_headless(
        {.first = 3, .samples = 4, .accumulates = true, .scene_changes = true, .time_frozen = true});
    CHECK(frozen.sample({.frame = 4, .sample = 0}).accumulated_since == 12);

    // A graph that accumulates nothing: one sample a frame, whatever was asked.
    const frame::HeadlessPlan once = frame::plan_headless(
        {.first = 3, .samples = 64, .accumulates = false, .scene_changes = true, .time_frozen = false});
    CHECK(once.samples == 1);
    CHECK(once.sample({.frame = 4, .sample = 0}).index == 4);
    CHECK(once.sample({.frame = 4, .sample = 0}).accumulated_since == 4);
}

TEST_CASE("the plans refuse what would wrap an index: frames out of order, a sample past the frame's, too many") {
    frame::LiveHistory live(frame::SceneMotion::still);
    CHECK(live.since(10, frame::View::changed) == 10);
    // Before the image's first: index - since would wrap.
    CHECK_THROWS_AS(live.since(9, frame::View::same), HistoryError);

    const frame::HeadlessPlan plan = frame::plan_headless(
        {.first = 0, .samples = 4, .accumulates = true, .scene_changes = false, .time_frozen = false});
    CHECK_THROWS_AS(plan.sample({.frame = 3, .sample = 4}), std::invalid_argument);
    CHECK(plan.sample({.frame = 3, .sample = 3}).index == 15);
    const std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(plan.sample({.frame = most / 4, .sample = 0}), std::overflow_error);
    CHECK(plan.sample({.frame = most / 4 - 1, .sample = 3}).index == most / 4 * 4 - 1);
    const frame::HeadlessPlan none = frame::plan_headless(
        {.first = 0, .samples = 0, .accumulates = true, .scene_changes = false, .time_frozen = false});
    CHECK_THROWS_AS(none.sample({.frame = 0, .sample = 0}), std::invalid_argument);
}
