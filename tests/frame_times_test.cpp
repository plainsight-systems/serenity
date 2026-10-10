// Frame times over a reporting period, driven with numbers.

#include <stdexcept>

#include <doctest/doctest.h>

#include "core/measurement/frame_times.h"

using serenity::frame::Seconds;
using serenity::measurement::FrameTimes;

TEST_CASE("a summary covers the frames of one period, then starts the next") {
    FrameTimes times(Seconds(1.0));
    CHECK_FALSE(times.take(Seconds(0.0)).has_value());  // starts the first period

    times.add(Seconds(10.000), Seconds(10.004));
    times.add(Seconds(10.010), Seconds(10.012));
    times.add(Seconds(10.020), Seconds(10.026));
    CHECK_FALSE(times.take(Seconds(0.5)).has_value());  // the period has not passed

    const auto summary = times.take(Seconds(1.0));
    REQUIRE(summary.has_value());
    CHECK(summary->frames == 3);
    CHECK(summary->mean.count() == doctest::Approx(0.004));
    CHECK(summary->shortest.count() == doctest::Approx(0.002));
    CHECK(summary->longest.count() == doctest::Approx(0.006));

    // The next period starts empty.
    CHECK_FALSE(times.take(Seconds(2.5)).has_value());
    times.add(Seconds(20.0), Seconds(20.001));
    const auto next = times.take(Seconds(2.6));
    REQUIRE(next.has_value());
    CHECK(next->frames == 1);
    CHECK(next->mean.count() == doctest::Approx(0.001));
}

TEST_CASE("a frame that ends before it begins is refused, and nothing is added") {
    FrameTimes times(Seconds(1.0));
    CHECK_FALSE(times.take(Seconds(0.0)).has_value());
    CHECK_THROWS_AS(times.add(Seconds(2.0), Seconds(1.0)), std::invalid_argument);
    CHECK_FALSE(times.take(Seconds(5.0)).has_value());  // no frame was added
}
