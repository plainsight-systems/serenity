// A frame, end to end, headless: scene text to schedule, schedule to the Metal
// renderer, the renderer into an offscreen image, the image read back. The
// window's path differs only in its target (metal/frame/renderer.h).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{64, 32};

frame::Schedule test_pattern_schedule() {
    return frame::parse_schedule("passes = [\"test_pattern\"]\n", "test");
}

struct Rig {
    metal::Device device;
    metal::Submission submission{device};
    metal::Offscreen target{device, submission, size};
    metal::Renderer renderer{device, submission, test_pattern_schedule(), nullptr};

    std::vector<std::uint8_t> render(double seconds, std::uint64_t index) {
        const auto sequence =
            metal::render_to_offscreen(submission, target, renderer, frame::FrameInputs{frame::Seconds(seconds), index, std::nullopt});
        submission.wait_until_complete(sequence);
        std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
        target.read_rgba(rgba);
        return rgba;
    }
};

// The test pattern's formula (test_pattern.h), as 8-bit values.
int expected(double value) {
    return static_cast<int>(std::lround(value * 255.0));
}

void check_pixel(const std::vector<std::uint8_t>& rgba, std::uint32_t x, std::uint32_t y, double seconds) {
    const std::size_t at = (std::size_t{y} * size.width + x) * 4;
    const double red = double(x) / (size.width - 1);
    const double green = double(y) / (size.height - 1);
    const double blue = 0.5 + 0.5 * std::sin(2.0 * std::numbers::pi * seconds / 4.0);
    INFO("pixel (" << x << ", " << y << ") at t = " << seconds);
    // Within one step of 8-bit storage and of the GPU's sin.
    CHECK(std::abs(rgba[at + 0] - expected(red)) <= 1);
    CHECK(std::abs(rgba[at + 1] - expected(green)) <= 1);
    CHECK(std::abs(rgba[at + 2] - expected(blue)) <= 1);
    CHECK(rgba[at + 3] == 255);
}

}  // namespace

TEST_CASE("the test pattern scene renders its formula") {
    Rig rig;
    for (double seconds : {1.0, 3.0, 0.5}) {
        const auto rgba = rig.render(seconds, 0);
        for (auto [x, y] : {std::pair{0u, 0u}, {63u, 0u}, {0u, 31u}, {63u, 31u}, {21u, 13u}}) {
            check_pixel(rgba, x, y, seconds);
        }
    }
}

TEST_CASE("time changes the image, and the same inputs give the same bytes") {
    Rig rig;
    const auto at_one = rig.render(1.0, 60);
    const auto at_three = rig.render(3.0, 180);
    const auto at_one_again = rig.render(1.0, 60);
    CHECK(at_one != at_three);
    CHECK(at_one == at_one_again);
}

TEST_CASE("frames in flight: each submission reads its own frame's constants") {
    // Four frames, each into its own image, at four times, committed without
    // waiting, so two are in flight at once and the ring's two slots are each
    // used twice. Had the frames shared one constants slot, an earlier frame
    // would show a later frame's time.
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, test_pattern_schedule(), nullptr);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    for (int i = 0; i < 4; ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
    }
    const double times[] = {1.0, 3.0, 0.5, 2.5};  // blue 1, 0, ~0.85, ~0.15: all distinct
    std::uint64_t last = 0;
    for (std::uint64_t i = 0; i < 4; ++i) {
        last = metal::render_to_offscreen(submission, *targets[i], renderer,
                                          frame::FrameInputs{frame::Seconds(times[i]), i, std::nullopt});
    }
    submission.wait_until_complete(last);  // in order, so every earlier one is done too
    std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
    for (std::size_t i = 0; i < 4; ++i) {
        targets[i]->read_rgba(rgba);
        check_pixel(rgba, 63, 31, times[i]);
        check_pixel(rgba, 21, 13, times[i]);
    }
}

TEST_CASE("finish() settles every frame, and nothing begins after it") {
    Rig rig;
    for (std::uint64_t i = 0; i < 3; ++i) {
        (void)metal::render_to_offscreen(rig.submission, rig.target, rig.renderer,
                                         frame::FrameInputs{frame::Seconds(0.0), i, std::nullopt});
    }
    rig.submission.finish();
    CHECK_THROWS_AS((void)rig.submission.begin(), metal::Error);
}

TEST_CASE("the submission protocol refuses misuse") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(submission.commit(), metal::Error);
    (void)submission.begin();
    CHECK_THROWS_AS((void)submission.begin(), metal::Error);
    CHECK_THROWS_AS(submission.wait_until_complete(0), metal::Error);
    submission.commit();
    submission.wait_until_complete(0);
    CHECK_THROWS_AS(submission.wait_until_complete(5), metal::Error);
}

TEST_CASE("an empty schedule is refused") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(metal::Renderer(device, submission, frame::Schedule{}, nullptr), metal::Error);
}

TEST_CASE("each submission is settled once, by whichever call comes first") {
    metal::Device device;
    metal::Submission submission(device);
    std::optional<metal::Completed> settled_by_begin;
    for (int i = 0; i < 3; ++i) {
        const metal::FrameSlot frame = submission.begin();
        if (i < 2) {
            CHECK_FALSE(frame.settled.has_value());  // its slot had not been used
        } else {
            settled_by_begin = frame.settled;  // submission 0 had used it
        }
        submission.commit();
    }
    REQUIRE(settled_by_begin.has_value());
    CHECK(settled_by_begin->sequence == 0);

    const auto waited = submission.wait_until_complete(2);
    REQUIRE(waited.has_value());
    CHECK(waited->sequence == 2);
    CHECK(waited->gpu_end >= waited->gpu_start);
    CHECK_FALSE(submission.wait_until_complete(2).has_value());  // already settled

    const std::vector<metal::Completed> rest = submission.finish();
    REQUIRE(rest.size() == 1);
    CHECK(rest[0].sequence == 1);
}
