// What the two programs check beyond their command lines' grammar
// (options_test.cpp): the window's time, its render scale, numbers read
// whole, and the headless renderer's output directory.

#include <filesystem>
#include <limits>
#include <vector>

#include <doctest/doctest.h>

#include "app/clock.h"
#include "app/options.h"
#include "headless/options.h"
#include "support/files.h"

TEST_CASE("window: a frame's time is its presentation instant less the first frame's") {
    using serenity::frame::Seconds;
    // Host-time instants as a display link gives them: far from 0, and
    // unevenly spaced, as a frame shown late is.
    serenity::app::PresentationClock clock;
    CHECK(clock.at(Seconds{86400.25}) == Seconds{0.0});
    CHECK(clock.at(Seconds{86400.25 + 0.0625}) == Seconds{0.0625});
    CHECK(clock.at(Seconds{86400.25 + 0.1875}) == Seconds{0.1875});
    // The origin is the first instant given, whatever it is.
    serenity::app::PresentationClock other;
    CHECK(other.at(Seconds{3.0}) == Seconds{0.0});
    CHECK(other.at(Seconds{3.5}) == Seconds{0.5});
}

TEST_CASE("window: a render scale outside (0, 1] is refused by render_size, as by parse") {
    using serenity::app::render_size;
    using serenity::frame::Extent;
    for (const double wrong : {0.0, -0.5, 1.5, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
        INFO("scale " << wrong);
        CHECK_THROWS_AS(render_size(Extent{3456, 2234}, wrong), serenity::app::OptionsError);
    }
}

TEST_CASE("both programs read a number whole: no sign, no space, nothing after it") {
    using serenity::app::OptionsError;
    for (const char* wrong : {"+0.5", " 0.5", "0.5 ", "0x1p-1"}) {
        INFO("--scale '" << wrong << "'");
        CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--graph", "g", "--scale", wrong}), OptionsError);
    }
    const auto headless = [](const char* option, const char* value) {
        return serenity::headless::parse(std::vector<const char*>{"--graph", "g", "--out", "o", option, value});
    };
    for (const char* wrong : {"+3", " 3", "3 ", "-0"}) {
        INFO("--frames '" << wrong << "'");
        CHECK_THROWS_AS(headless("--frames", wrong), serenity::headless::OptionsError);
    }
    for (const char* wrong : {"+0.5", " 0.5"}) {
        INFO("--step '" << wrong << "'");
        CHECK_THROWS_AS(headless("--step", wrong), serenity::headless::OptionsError);
    }
}

TEST_CASE("headless: the output directory holds this run's frames and nothing else") {
    using serenity::headless::prepare_output;
    using serenity::headless::OptionsError;
    const serenity::tests::ScratchDirectory scratch("programs-test-out");
    const std::filesystem::path out = scratch.path() / "parent" / "frames";

    CHECK_NOTHROW(prepare_output(out));  // made, with its parents
    CHECK(std::filesystem::is_directory(out));
    CHECK_NOTHROW(prepare_output(out));  // there and empty: taken

    serenity::tests::write_bytes(out / "frame-000000.png", "an earlier run's frame");
    CHECK_THROWS_AS(prepare_output(out), OptionsError);

    // A file in the directory's place: OptionsError, as the header states,
    // not the file system's own error from trying to make it.
    const std::filesystem::path file = scratch / "a-file";
    serenity::tests::write_bytes(file, "not a directory");
    CHECK_THROWS_AS(prepare_output(file), OptionsError);
}
