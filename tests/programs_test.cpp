// What the two programs check beyond their command lines' grammar
// (options_test.cpp): the window's clock, its render scale, numbers read
// whole, and the headless renderer's output directory.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "app/clock.h"
#include "app/options.h"
#include "headless/options.h"

namespace {

// A directory of the test's own under the system's temporary directory,
// removed when the test ends.
struct ScratchDirectory {
    std::filesystem::path path;

    explicit ScratchDirectory(const std::string& name)
        : path(std::filesystem::temp_directory_path() / ("serenity-programs-test-" + name)) {
        std::filesystem::remove_all(path);
    }
    ~ScratchDirectory() { std::filesystem::remove_all(path); }
    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;
};

}  // namespace

TEST_CASE("window: the clock counts from its construction and never runs backwards") {
    const serenity::app::Clock clock;
    serenity::frame::Seconds last = clock.elapsed();
    CHECK(last.count() >= 0.0);
    for (int i = 0; i < 1000; ++i) {
        const serenity::frame::Seconds now = clock.elapsed();
        CHECK(now >= last);
        last = now;
    }
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
    for (const char* wrong : {"+3", " 3", "3 ", "-0"}) {
        INFO("--frames '" << wrong << "'");
        CHECK_THROWS_AS(serenity::headless::parse(std::vector<const char*>{"--graph", "g", "--out", "o", "--frames", wrong}),
                        serenity::headless::OptionsError);
    }
    for (const char* wrong : {"+0.5", " 0.5"}) {
        INFO("--step '" << wrong << "'");
        CHECK_THROWS_AS(serenity::headless::parse(std::vector<const char*>{"--graph", "g", "--out", "o", "--step", wrong}),
                        serenity::headless::OptionsError);
    }
}

TEST_CASE("headless: a way of writing with no rule is an error, not a frame written") {
    using serenity::headless::Write;
    // An enumeration with no fixed type holds the values its enumerators'
    // bits span: 0 to 3 here, 3 naming none.
    CHECK_THROWS_AS(serenity::headless::written(static_cast<Write>(3), {.after_first = 0, .frames = 1}),
                    std::logic_error);
}

TEST_CASE("headless: the output directory holds this run's frames and nothing else") {
    using serenity::headless::prepare_output;
    const ScratchDirectory scratch("out");
    const std::filesystem::path out = scratch.path / "frames";

    CHECK_NOTHROW(prepare_output(out));  // made, with its parents
    CHECK(std::filesystem::is_directory(out));
    CHECK_NOTHROW(prepare_output(out));  // there and empty: taken

    std::ofstream(out / "frame-000000.png") << "an earlier run's frame";
    CHECK_THROWS_AS(prepare_output(out), serenity::headless::OptionsError);

    const std::filesystem::path file = scratch.path / "a-file";
    std::ofstream(file) << "not a directory";
    CHECK_THROWS(prepare_output(file));
}
