// The camera's jitter (core/sampling/jitter.h): Halton(2, 3) of 32 phases,
// pinned to the table training/test_training.py and
// tests/test_make_denoiser_dataset.py hold their own copies of the formula
// to; its 32 phases cover the pixel; and which graphs take it
// (core/frame/schedule.h, jitters).

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <utility>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"
#include "core/sampling/jitter.h"
#include "support/schedules.h"

using namespace serenity;

TEST_CASE("the camera's jitter is Halton(2, 3) from k = 1, its 32 phases repeating, the same as its copies'") {
    // index, x, y: the table the Python copies are held to.
    constexpr std::array<std::pair<std::uint64_t, std::array<double, 2>>, 7> table = {{
        {0, {0.0, -1.0 / 6.0}},
        {1, {-0.25, 1.0 / 6.0}},
        {2, {0.25, -7.0 / 18.0}},
        {31, {-0.484375, 0.2901234567901234}},
        {32, {0.0, -1.0 / 6.0}},
        {7200, {0.0, -1.0 / 6.0}},
        {7201, {-0.25, 1.0 / 6.0}},
    }};
    for (const auto& [index, expected] : table) {
        const sampling::CameraJitter j = sampling::camera_jitter(index);
        INFO("frame " << index);
        CHECK(j.x == doctest::Approx(expected[0]).epsilon(1e-7));
        CHECK(j.y == doctest::Approx(expected[1]).epsilon(1e-7));
    }
    static_assert(sampling::camera_jitter(5).x == sampling::camera_jitter(5 + contracts::jitter_phases).x);
}

TEST_CASE("the jitter's 32 phases cover the pixel: each x in its own 1/32, each of the first 27 y in its own 1/27") {
    std::set<int> x_strata;
    std::set<int> y_strata;
    for (std::uint64_t i = 0; i < contracts::jitter_phases; ++i) {
        const sampling::CameraJitter j = sampling::camera_jitter(i);
        CHECK(j.x >= -0.5f);
        CHECK(j.x < 0.5f);
        CHECK(j.y >= -0.5f);
        CHECK(j.y < 0.5f);
        x_strata.insert(static_cast<int>((static_cast<double>(j.x) + 0.5) * 32.0));
        if (i < 27) {
            y_strata.insert(static_cast<int>((static_cast<double>(j.y) + 0.5) * 27.0 + 1e-4));  // m / 27 in float, a little under
        }
    }
    CHECK(x_strata.size() == contracts::jitter_phases);
    CHECK(y_strata.size() == 27);
}

TEST_CASE("ReSTIR PT's graphs take the jitter; the path tracer and the features alone do not") {
    using frame::PassKind;
    std::vector<PassKind> light = tests::restir_pt_passes;
    light.push_back(PassKind::features);
    light.push_back(PassKind::tone_map);
    frame::Schedule restir{light, tests::look};
    restir.restir_pt = tests::written_restir_pt(light);
    CHECK(frame::jitters(restir));
    CHECK_FALSE(frame::jitters(frame::Schedule{{PassKind::path, PassKind::tone_map}, tests::look}));
    CHECK_FALSE(frame::jitters(frame::Schedule{{PassKind::features}, std::nullopt}));
}
