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
    const auto defaults = headless({"--scene", "s.toml", "--out", "o"});
    CHECK(defaults.frames == 1);
    CHECK(defaults.first == 0);
    CHECK(defaults.step.count() == doctest::Approx(1.0 / 60.0));
    CHECK(defaults.size == serenity::frame::Extent{1920, 1080});
    CHECK(defaults.scene == "s.toml");
    CHECK(defaults.out == "o");

    const auto all = headless({"--scene", "s.toml", "--out", "o", "--frames", "120", "--first", "7", "--step", "0.5",
                               "--size", "640x360"});
    CHECK(all.frames == 120);
    CHECK(all.first == 7);
    CHECK(all.step.count() == 0.5);
    CHECK(all.size == serenity::frame::Extent{640, 360});
}

TEST_CASE("headless: every mistake is refused by name") {
    CHECK(contains(headless_error({"--out", "o"}), "missing --scene"));
    CHECK(contains(headless_error({"--scene", "s"}), "missing --out"));
    CHECK(contains(headless_error({"--scene"}), "--scene needs a value"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--frames", "0"}), "at least 1"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--frames", "-3"}), "whole number"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--frames", "3x"}), "whole number"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--step", "0"}), "positive"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--step", "inf"}), "positive"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--size", "640"}), "WIDTHxHEIGHT"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--size", "0x360"}), "at least 1"));
    CHECK(contains(headless_error({"--scene", "s", "--out", "o", "--fast"}), "unknown option '--fast'"));
}

TEST_CASE("window: the scene is required, and nothing else is accepted") {
    CHECK(serenity::app::parse(std::vector<const char*>{"--scene", "s.toml"}).scene == "s.toml");
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{}), serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--scene"}), serenity::app::OptionsError);
    CHECK_THROWS_AS(serenity::app::parse(std::vector<const char*>{"--scene", "s", "--size", "1x1"}),
                    serenity::app::OptionsError);
}
