// The two command lines: what they accept, and every mistake refused by name.

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "app/options.h"
#include "core/frame/frame_inputs.h"
#include "core/output/image_format.h"
#include "headless/options.h"
#include "support/text.h"

using serenity::tests::contains;

namespace {

// The headless renderer's command line: the two options it requires, then
// `more`.
std::vector<const char*> headless_args(std::initializer_list<const char*> more) {
    std::vector<const char*> args = {"--graph", "s", "--out", "o"};
    args.insert(args.end(), more);
    return args;
}

serenity::headless::Options headless(std::initializer_list<const char*> more) {
    return serenity::headless::parse(headless_args(more));
}

// What the headless renderer's parse() refuses `args` with, or "".
std::string headless_error(const std::vector<const char*>& args) {
    return serenity::tests::error_of<serenity::headless::OptionsError>(
        [&] { return serenity::headless::parse(args); });
}

std::string headless_error(std::initializer_list<const char*> more) {
    return headless_error(headless_args(more));
}

// The last sample index whose random numbers are its own, 2^32 - 1
// (headless/options.h), the most a 64-bit count holds, and an image's most
// frames, written out.
constexpr std::uint64_t last_sample = (std::uint64_t{1} << 32) - 1;
const std::string last_index = std::to_string(last_sample);
const std::string past_last_index = std::to_string(last_sample + 1);
const std::string largest = std::to_string(std::numeric_limits<std::uint64_t>::max());
const std::string most_frames = std::to_string(serenity::frame::max_accumulated_frames);
const std::string past_most_frames = std::to_string(serenity::frame::max_accumulated_frames + 1);

}  // namespace

TEST_CASE("headless: defaults, and every option read") {
    const auto defaults = serenity::headless::parse(std::vector<const char*>{"--graph", "s.toml", "--out", "o"});
    CHECK(defaults.frames == 1);
    CHECK(defaults.first == 0);
    CHECK(defaults.step.count() == doctest::Approx(1.0 / 60.0).scale(0).epsilon(1e-6));
    CHECK(defaults.size == serenity::frame::Extent{1920, 1080});
    CHECK(defaults.graph == "s.toml");
    CHECK(defaults.out == "o");

    const auto all = headless({"--frames", "120", "--first", "7", "--step", "0.5", "--size", "640x360"});
    CHECK(all.frames == 120);
    CHECK(all.first == 7);
    CHECK(all.step.count() == 0.5);
    CHECK(all.size == serenity::frame::Extent{640, 360});
}

TEST_CASE("headless: every mistake is refused by name") {
    CHECK(contains(headless_error(std::vector<const char*>{"--out", "o"}), "missing --graph"));
    CHECK(contains(headless_error(std::vector<const char*>{"--graph", "s"}), "missing --out"));
    CHECK(contains(headless_error(std::vector<const char*>{"--graph"}), "--graph needs a value"));
    CHECK(contains(headless_error({"--frames", "0"}), "at least 1"));
    CHECK(contains(headless_error({"--frames", "-3"}), "whole number"));
    CHECK(contains(headless_error({"--frames", "3x"}), "whole number"));
    CHECK(contains(headless_error({"--step", "0"}), "positive"));
    CHECK(contains(headless_error({"--step", "inf"}), "positive"));
    CHECK(contains(headless_error({"--size", "640"}), "WIDTHxHEIGHT"));
    CHECK(contains(headless_error({"--size", "0x360"}), "at least 1"));
    CHECK(contains(headless_error({"--fast"}), "unknown option '--fast'"));
    CHECK(contains(headless_error({"--first", last_index.c_str(), "--frames", "2"}), "past sample index 2^32 - 1"));
}

TEST_CASE("headless: the last sample index there can be, 2^32 - 1, is reachable, and one past it is not") {
    // One sample a frame: frame 2^32 - 1 is the last.
    const auto last = headless({"--first", last_index.c_str(), "--frames", "1"});
    CHECK(last.first == last_sample);
    CHECK(last.frames == 1);
    CHECK(contains(headless_error({"--first", past_last_index.c_str()}), "past sample index 2^32 - 1"));
    // From frame 0, 2^32 frames end at the last; one more is past it.
    const std::string all_frames = std::to_string(last_sample + 1);
    CHECK(headless({"--frames", all_frames.c_str()}).frames == last_sample + 1);
    const std::string one_more = std::to_string(last_sample + 2);
    CHECK(contains(headless_error({"--frames", one_more.c_str()}), "past sample index"));
    // Numbers that would wrap a 64-bit sum or product are refused, not
    // wrapped into range (ES.103).
    CHECK(contains(headless_error({"--first", largest.c_str()}), "past sample index"));
    CHECK(contains(headless_error({"--first", "1", "--frames", largest.c_str()}), "past sample index"));
}

TEST_CASE("headless: samples per frame, from 1 to what an image holds, and the last sample reachable") {
    CHECK(headless({}).samples == 1);
    CHECK(headless({"--samples", "64"}).samples == 64);
    CHECK(headless({"--samples", most_frames.c_str()}).samples == serenity::frame::max_accumulated_frames);
    CHECK(contains(headless_error({"--samples", "0"}), "--samples must be from 1"));
    CHECK(contains(headless_error({"--samples", past_most_frames.c_str()}), "the most an image holds"));
    // The last frame's last sample, (first + frames) x N - 1, at most
    // 2^32 - 1: at N = 1024, frames first .. 2^22 - 1 exactly reach it, and
    // a frame more is past it. A reference's batch k at 1024 samples is
    // frame k, so 2^22 batches, and no more, have numbers of their own.
    const std::uint64_t frames_at_1024 = std::uint64_t{1} << 22;
    const std::string last_first = std::to_string(frames_at_1024 - 1);
    const std::string past_first = std::to_string(frames_at_1024);
    CHECK(headless({"--first", last_first.c_str(), "--samples", "1024"}).first == frames_at_1024 - 1);
    CHECK(contains(headless_error({"--first", past_first.c_str(), "--samples", "1024"}),
                   "--first 4194304 plus --frames 1, at --samples 1024 per frame, is past sample index 2^32 - 1"));
    CHECK(headless({"--first", "4194300", "--frames", "4", "--samples", "1024"}).frames == 4);
    CHECK(contains(headless_error({"--first", "4194300", "--frames", "5", "--samples", "1024"}), "past sample index"));
    // The most samples an image holds, from frame 0: 256 frames of them
    // reach (256 x (2^24 - 1)) - 1, inside the bound; the product is not
    // what wraps here, and is still checked.
    CHECK(headless({"--frames", "256", "--samples", most_frames.c_str()}).frames == 256);
    CHECK(contains(headless_error({"--frames", "257", "--samples", most_frames.c_str()}), "past sample index"));
    // The product exactly at the bound and one past it: 2^31 frames of 2
    // samples end at index 2^32 - 1; 6700417 frames of 641, 2^32 + 1
    // samples (its factors), end one past.
    CHECK(headless({"--frames", "2147483648", "--samples", "2"}).frames == 2147483648u);
    CHECK(contains(headless_error({"--frames", "6700417", "--samples", "641"}), "past sample index"));
    CHECK(headless({"--frames", "6700416", "--samples", "641"}).frames == 6700416);
}

TEST_CASE("headless: what a written frame is, png by default or pfm, and nothing else") {
    using serenity::output::ImageFormat;
    CHECK(headless({}).format == ImageFormat::png);
    CHECK(headless({"--format", "png"}).format == ImageFormat::png);
    CHECK(headless({"--format", "pfm"}).format == ImageFormat::pfm);
    for (const char* wrong : {"PFM", "exr", "", " pfm", "pfm "}) {
        INFO("--format '" << wrong << "'");
        CHECK(contains(headless_error({"--format", wrong}), "--format needs png or pfm"));
    }
    CHECK(contains(headless_error({"--format"}), "--format needs a value"));
}

TEST_CASE("output: the kinds a frame is written as, by name and by extension") {
    using serenity::output::extension;
    using serenity::output::image_format_named;
    using serenity::output::ImageFormat;
    CHECK(image_format_named("png") == ImageFormat::png);
    CHECK(image_format_named("pfm") == ImageFormat::pfm);
    CHECK_FALSE(image_format_named("Png").has_value());
    CHECK_FALSE(image_format_named("").has_value());
    CHECK(extension(ImageFormat::png) == "png");
    CHECK(extension(ImageFormat::pfm) == "pfm");
    // extension()'s refusal of a value no kind names is not reached here:
    // the enumeration's two kinds span one bit, so no value of it is
    // unnamed, and casting one in would be undefined behavior.
}

TEST_CASE("headless: a scene is optional") {
    CHECK(headless({}).scene.empty());
    CHECK(headless({"--scene", "s.toml"}).scene == "s.toml");
    CHECK(contains(headless_error({"--scene"}), "--scene needs a value"));
}

TEST_CASE("window: the graph is required, the scene optional, and nothing else is accepted") {
    using serenity::app::OptionsError;
    using serenity::app::parse;
    CHECK(parse(std::vector<const char*>{"--graph", "s.toml"}).graph == "s.toml");
    CHECK(parse(std::vector<const char*>{"--graph", "g.toml"}).scene.empty());
    const auto both = parse(std::vector<const char*>{"--graph", "g.toml", "--scene", "s.toml"});
    CHECK(both.graph == "g.toml");
    CHECK(both.scene == "s.toml");
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--scene", "s.toml"}), OptionsError);
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "g", "--scene"}), OptionsError);
    CHECK_THROWS_AS(parse(std::vector<const char*>{}), OptionsError);
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph"}), OptionsError);
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "s", "--size", "1x1"}), OptionsError);
}

TEST_CASE("window: a render scale in (0, 1], and the size it gives") {
    using serenity::app::OptionsError;
    using serenity::app::parse;
    using serenity::app::render_size;
    using serenity::frame::Extent;
    CHECK(parse(std::vector<const char*>{"--graph", "g"}).scale == 1.0);
    CHECK(parse(std::vector<const char*>{"--graph", "g", "--scale", "0.5"}).scale == 0.5);
    for (const char* wrong : {"0", "-0.5", "1.5", "nan", "inf", "half", "0.5x"}) {
        INFO("--scale " << wrong);
        CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "g", "--scale", wrong}), OptionsError);
    }
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "g", "--scale"}), OptionsError);

    CHECK(render_size(Extent{3456, 2234}, 1.0) == Extent{3456, 2234});
    CHECK(render_size(Extent{3456, 2234}, 0.5) == Extent{1728, 1117});
    CHECK(render_size(Extent{2560, 1600}, 0.33) == Extent{845, 528});
    CHECK(render_size(Extent{3, 1}, 0.01) == Extent{1, 1});  // never empty
}

TEST_CASE("headless: which frames are written, and frozen time") {
    using serenity::headless::Write;
    using serenity::headless::written;
    CHECK(headless({}).write == Write::all);
    CHECK(headless({"--write", "last"}).write == Write::last);
    CHECK(headless({"--write", "doubling"}).write == Write::doubling);
    CHECK(contains(headless_error({"--write", "some"}), "all, last or doubling"));

    // Of 10 frames: all; the last; the 1st, 2nd, 4th, 8th and the last.
    constexpr std::uint64_t frames = 10;
    std::vector<std::uint64_t> all;
    std::vector<std::uint64_t> last;
    std::vector<std::uint64_t> doubling;
    for (std::uint64_t n = 0; n < frames; ++n) {
        if (written(Write::all, {.after_first = n, .frames = frames})) {
            all.push_back(n);
        }
        if (written(Write::last, {.after_first = n, .frames = frames})) {
            last.push_back(n);
        }
        if (written(Write::doubling, {.after_first = n, .frames = frames})) {
            doubling.push_back(n);
        }
    }
    CHECK(all.size() == frames);
    CHECK(last == std::vector<std::uint64_t>{frames - 1});
    CHECK(doubling == std::vector<std::uint64_t>{0, 1, 3, 7, 9});

    CHECK_FALSE(headless({}).time.has_value());
    const auto frozen = headless({"--time", "2.5"});
    REQUIRE(frozen.time.has_value());
    CHECK(frozen.time->count() == 2.5);
    CHECK(headless({"--time", "0"}).time->count() == 0.0);
    CHECK(contains(headless_error({"--time", "-1"}), "0 or more"));
    CHECK(contains(headless_error({"--time", "nan"}), "0 or more"));
    CHECK(contains(headless_error({"--time", "1", "--step", "0.5"}), "exclusive"));
}
