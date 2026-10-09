// The two command lines: what they accept, and every mistake refused by name.

#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "app/options.h"
#include "headless/options.h"

namespace {

serenity::headless::Options headless(std::vector<const char*> args) {
    return serenity::headless::parse(args);
}

std::string headless_error(std::vector<const char*> args) {
    try {
        (void)serenity::headless::parse(args);
    } catch (const serenity::headless::Error& error) {
        return error.what();
    }
    return "";
}

bool contains(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TEST_CASE("headless: defaults, and every option read") {
    const auto defaults = headless({"--graph", "s.toml", "--out", "o"});
    CHECK(defaults.frames == 1);
    CHECK(defaults.first == 0);
    CHECK(defaults.step.count() == doctest::Approx(1.0 / 60.0));
    CHECK(defaults.size == serenity::frame::Extent{1920, 1080});
    CHECK(defaults.graph == "s.toml");
    CHECK(defaults.out == "o");

    const auto all = headless({"--graph", "s.toml", "--out", "o", "--frames", "120", "--first", "7", "--step", "0.5",
                               "--size", "640x360"});
    CHECK(all.frames == 120);
    CHECK(all.first == 7);
    CHECK(all.step.count() == 0.5);
    CHECK(all.size == serenity::frame::Extent{640, 360});
}

TEST_CASE("headless: every mistake is refused by name") {
    CHECK(contains(headless_error({"--out", "o"}), "missing --graph"));
    CHECK(contains(headless_error({"--graph", "s"}), "missing --out"));
    CHECK(contains(headless_error({"--graph"}), "--graph needs a value"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--frames", "0"}), "at least 1"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--frames", "-3"}), "whole number"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--frames", "3x"}), "whole number"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--step", "0"}), "positive"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--step", "inf"}), "positive"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--size", "640"}), "WIDTHxHEIGHT"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--size", "0x360"}), "at least 1"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--fast"}), "unknown option '--fast'"));
    CHECK(contains(headless_error({"--graph", "s", "--out", "o", "--first", "18446744073709551615", "--frames", "2"}),
                   "past the last frame"));
}

TEST_CASE("headless: the last frame there can be is reachable") {
    const auto last = headless({"--graph", "s", "--out", "o", "--first", "18446744073709551615", "--frames", "1"});
    CHECK(last.first == 18446744073709551615ull);
    CHECK(last.frames == 1);
}

TEST_CASE("headless: a scene is optional") {
    CHECK(headless({"--graph", "g.toml", "--out", "o"}).scene.empty());
    CHECK(headless({"--graph", "g.toml", "--scene", "s.toml", "--out", "o"}).scene == "s.toml");
    CHECK(contains(headless_error({"--graph", "g", "--out", "o", "--scene"}), "--scene needs a value"));
}

TEST_CASE("window: the graph is required, the scene optional, and nothing else is accepted") {
    CHECK(serenity::app::parse(std::vector<const char*>{"--graph", "s.toml"}).graph == "s.toml");
    CHECK(serenity::app::parse(std::vector<const char*>{"--graph", "g.toml"}).scene.empty());
    const auto both = serenity::app::parse(std::vector<const char*>{"--graph", "g.toml", "--scene", "s.toml"});
    CHECK(both.graph == "g.toml");
    CHECK(both.scene == "s.toml");
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--scene", "s.toml"}), serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--graph", "g", "--scene"}),
                    serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{}), serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--graph"}), serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--graph", "s", "--size", "1x1"}),
                    serenity::app::OptionsError);
}

TEST_CASE("window: a render scale in (0, 1], and the size it gives") {
    using serenity::app::parse;
    using serenity::app::render_size;
    using serenity::frame::Extent;
    CHECK(parse(std::vector<const char*>{"--graph", "g"}).scale == 1.0);
    CHECK(parse(std::vector<const char*>{"--graph", "g", "--scale", "0.5"}).scale == 0.5);
    for (const char* wrong : {"0", "-0.5", "1.5", "nan", "inf", "half", "0.5x"}) {
        INFO("--scale " << wrong);
        CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "g", "--scale", wrong}),
                        serenity::app::OptionsError);
    }
    CHECK_THROWS_AS(parse(std::vector<const char*>{"--graph", "g", "--scale"}), serenity::app::OptionsError);

    CHECK(render_size(Extent{3456, 2234}, 1.0) == Extent{3456, 2234});
    CHECK(render_size(Extent{3456, 2234}, 0.5) == Extent{1728, 1117});
    CHECK(render_size(Extent{2560, 1600}, 0.33) == Extent{845, 528});
    CHECK(render_size(Extent{3, 1}, 0.01) == Extent{1, 1});  // never empty
}
