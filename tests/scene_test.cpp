// Scene files: the schedule is read as written, and every mistake is an
// error naming the file and the line, never a default.

#include <string>

#include <doctest/doctest.h>

#include "core/scene/scene.h"

using serenity::frame::PassKind;
using serenity::scene::Error;
using serenity::scene::load;
using serenity::scene::parse;

namespace {

// The error parse() throws for `text`, or "" if it throws none.
std::string error_for(const char* text) {
    try {
        (void)parse(text, "scene.toml");
    } catch (const Error& error) {
        return error.what();
    }
    return "";
}

bool contains(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TEST_CASE("the schedule is read in order") {
    const auto scene = parse("[frame]\npasses = [\"test_pattern\", \"test_pattern\"]\n", "scene.toml");
    REQUIRE(scene.schedule.passes.size() == 2);
    CHECK(scene.schedule.passes[0] == PassKind::test_pattern);
    CHECK(scene.schedule.passes[1] == PassKind::test_pattern);
}

TEST_CASE("comments are allowed") {
    const auto scene = parse("# a scene\n[frame]\npasses = [\"test_pattern\"]  # the only one\n", "scene.toml");
    CHECK(scene.schedule.passes.size() == 1);
}

TEST_CASE("an unknown pass is an error that names it, its line, and the known passes") {
    const std::string error = error_for("[frame]\npasses = [\n  \"test_pattern\",\n  \"tset_pattern\",\n]\n");
    CHECK(contains(error, "scene.toml:4:"));
    CHECK(contains(error, "'tset_pattern'"));
    CHECK(contains(error, "known passes: test_pattern"));
}

TEST_CASE("every structural mistake is an error") {
    CHECK(contains(error_for(""), "missing [frame]"));
    CHECK(contains(error_for("frame = 1\n"), "'frame' must be a table"));
    CHECK(contains(error_for("[frame]\n"), "has no 'passes'"));
    CHECK(contains(error_for("[frame]\npasses = []\n"), "non-empty array"));
    CHECK(contains(error_for("[frame]\npasses = \"test_pattern\"\n"), "non-empty array"));
    CHECK(contains(error_for("[frame]\npasses = [1]\n"), "must be a name"));
}

TEST_CASE("a key nobody reads is an error, not ignored") {
    CHECK(contains(error_for("[frame]\npasses = [\"test_pattern\"]\npass = [\"x\"]\n"), "scene.toml:3: unknown key 'pass'"));
    CHECK(contains(error_for("[frame]\npasses = [\"test_pattern\"]\n[camera]\n"), "unknown key 'camera'"));
}

TEST_CASE("malformed TOML is an error with its line") {
    CHECK(contains(error_for("[frame]\npasses = [\"test_pattern\"\n"), "scene.toml:"));
}

TEST_CASE("the test pattern scene in scenes/ reads") {
    const auto scene = load(SERENITY_SCENES_DIR "/test_pattern.toml");
    REQUIRE(scene.schedule.passes.size() == 1);
    CHECK(scene.schedule.passes[0] == PassKind::test_pattern);
}

TEST_CASE("a missing file is an error that names it") {
    try {
        (void)load("no/such/scene.toml");
        FAIL("expected an error");
    } catch (const Error& error) {
        CHECK(contains(error.what(), "no/such/scene.toml"));
    }
}
