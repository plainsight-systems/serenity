// A frame, end to end, headless: scene text to schedule, schedule to the Metal
// renderer, the renderer into an offscreen image, the image read back. The
// window's path differs only in its target (metal/frame/renderer.h).

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/references.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{64, 32};

// The test pattern's blue pulses once every this many seconds
// (metal/passes/test_pattern/test_pattern.h).
constexpr double pulse_period = 4.0;

frame::Schedule test_pattern_schedule() {
    return frame::parse_schedule("passes = [\"test_pattern\"]\n", "test");
}

frame::FrameInputs at(double seconds, std::uint64_t index) {
    return tests::frame_at(index, index, seconds, std::nullopt);
}

struct Rig {
    metal::Device device;
    metal::Submission submission{device};
    metal::Offscreen target{device, submission, size};
    metal::Renderer renderer{device, submission, test_pattern_schedule(), nullptr};

    std::vector<std::uint8_t> render(double seconds, std::uint64_t index) {
        (void)submission.wait_until_complete(
            metal::render_to_offscreen(submission, target, renderer, at(seconds, index)));
        return tests::read_back(target);
    }
};

// The test pattern's formula (test_pattern.h), as 8-bit values.
int expected(double value) {
    return static_cast<int>(std::lround(value * 255.0));
}

void check_pixel(const std::vector<std::uint8_t>& rgba, tests::Pixel pixel, double seconds) {
    const std::size_t at = (std::size_t{pixel.y} * size.width + pixel.x) * 4;
    const double red = static_cast<double>(pixel.x) / (size.width - 1);
    const double green = static_cast<double>(pixel.y) / (size.height - 1);
    const double blue = 0.5 + 0.5 * std::sin(2.0 * std::numbers::pi * seconds / pulse_period);
    INFO("pixel (" << pixel.x << ", " << pixel.y << ") at t = " << seconds);
    // Within one step of 8-bit storage and of the GPU's sin.
    CHECK(std::abs(rgba[at + 0] - expected(red)) <= 1);
    CHECK(std::abs(rgba[at + 1] - expected(green)) <= 1);
    CHECK(std::abs(rgba[at + 2] - expected(blue)) <= 1);
    CHECK(rgba[at + 3] == 255);
}

// The corners and a pixel inside.
constexpr std::array<tests::Pixel, 5> probed{
    {{0, 0}, {size.width - 1, 0}, {0, size.height - 1}, {size.width - 1, size.height - 1}, {21, 13}}};

}  // namespace

TEST_CASE("the test pattern scene renders its formula") {
    Rig rig;
    for (const double seconds : {1.0, 3.0, 0.5}) {
        const auto rgba = rig.render(seconds, 0);
        for (const tests::Pixel pixel : probed) {
            check_pixel(rgba, pixel, seconds);
        }
    }
}

TEST_CASE("time changes the image, and the same inputs give the same bytes") {
    Rig rig;
    const auto at_one = rig.render(1.0, 60);
    const auto at_three = rig.render(3.0, 180);
    const auto at_one_again = rig.render(1.0, 60);
    CHECK(tests::differing(at_one, at_three) > 0);
    CHECK(tests::differing(at_one, at_one_again) == 0);
}

TEST_CASE("frames in flight: each submission reads its own frame's constants") {
    // Four frames, each into its own image, at four times, committed without
    // waiting, so two are in flight at once and the ring's two slots are each
    // used twice. Had the frames shared one constants slot, an earlier frame
    // would show a later frame's time.
    const std::array<double, 4> times{1.0, 3.0, 0.5, 2.5};  // blue 1, 0, ~0.85, ~0.15: all distinct
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, test_pattern_schedule(), nullptr);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    std::uint64_t last = 0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
        last = metal::render_to_offscreen(submission, *targets.back(), renderer, at(times[i], i));
    }
    (void)submission.wait_until_complete(last);  // in order, so every earlier one is done too
    for (std::size_t i = 0; i < times.size(); ++i) {
        const std::vector<std::uint8_t> rgba = tests::read_back(*targets[i]);
        check_pixel(rgba, probed[3], times[i]);
        check_pixel(rgba, probed[4], times[i]);
    }
}

TEST_CASE("finish() settles every frame, and nothing begins after it") {
    Rig rig;
    for (std::uint64_t i = 0; i < 3; ++i) {
        (void)metal::render_to_offscreen(rig.submission, rig.target, rig.renderer, at(0.0, i));
    }
    (void)rig.submission.finish();
    CHECK_THROWS_AS((void)rig.submission.begin(), metal::MetalError);
}

TEST_CASE("the submission protocol refuses misuse") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(submission.commit(), metal::MetalError);
    (void)submission.begin();
    CHECK_THROWS_AS((void)submission.begin(), metal::MetalError);
    CHECK_THROWS_AS((void)submission.wait_until_complete(0), metal::MetalError);
    submission.commit();
    (void)submission.wait_until_complete(0);
    CHECK_THROWS_AS((void)submission.wait_until_complete(5), metal::MetalError);
}

TEST_CASE("an empty schedule is refused") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(metal::Renderer(device, submission, frame::Schedule{}, nullptr), metal::MetalError);
}

TEST_CASE("each submission is settled once, by whichever call comes first") {
    metal::Device device;
    metal::Submission submission(device);
    std::optional<metal::Completed> settled_by_begin;
    for (int i = 0; i < 3; ++i) {
        const metal::FrameSlot slot = submission.begin();
        if (i < 2) {
            CHECK_FALSE(slot.settled.has_value());  // its slot had not been used
        } else {
            settled_by_begin = slot.settled;  // submission 0 had used it
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
