// Frame graph files: the passes are read as written, and every mistake is an
// error naming the file and the line, never a default.

#include <string>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"

using serenity::frame::GraphFileError;
using serenity::frame::load_schedule;
using serenity::frame::parse_schedule;
using serenity::frame::PassKind;

namespace {

// The error parse_schedule() throws for `text`, or "" if it throws none.
std::string error_for(const char* text) {
    try {
        (void)parse_schedule(text, "graph.toml");
    } catch (const GraphFileError& error) {
        return error.what();
    }
    return "";
}

bool contains(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TEST_CASE("the passes are read in order") {
    const auto schedule = parse_schedule("passes = [\"test_pattern\", \"test_pattern\"]\n", "graph.toml");
    REQUIRE(schedule.passes.size() == 2);
    CHECK(schedule.passes[0] == PassKind::test_pattern);
    CHECK(schedule.passes[1] == PassKind::test_pattern);
}

TEST_CASE("comments are allowed") {
    const auto schedule = parse_schedule("# a graph\npasses = [\"test_pattern\"]  # the only one\n", "graph.toml");
    CHECK(schedule.passes.size() == 1);
}

TEST_CASE("an unknown pass is an error that names it, its line, and the known passes") {
    const std::string error = error_for("passes = [\n  \"test_pattern\",\n  \"tset_pattern\",\n]\n");
    CHECK(contains(error, "graph.toml:3:"));
    CHECK(contains(error, "'tset_pattern'"));
    CHECK(contains(error, "known passes: test_pattern"));
}

TEST_CASE("every structural mistake is an error") {
    CHECK(contains(error_for(""), "no 'passes'"));
    CHECK(contains(error_for("passes = []\n"), "non-empty array"));
    CHECK(contains(error_for("passes = \"test_pattern\"\n"), "non-empty array"));
    CHECK(contains(error_for("passes = [1]\n"), "must be a name"));
}

TEST_CASE("a key nobody reads is an error, not ignored") {
    CHECK(contains(error_for("passes = [\"test_pattern\"]\npass = [\"x\"]\n"), "graph.toml:2: unknown key 'pass'"));
    CHECK(contains(error_for("passes = [\"test_pattern\"]\n[camera]\n"), "unknown key 'camera'"));
}

TEST_CASE("malformed TOML is an error with its line") {
    CHECK(contains(error_for("passes = [\"test_pattern\"\n"), "graph.toml:"));
}

TEST_CASE("the test pattern graph in graphs/ reads") {
    const auto schedule = load_schedule(SERENITY_GRAPHS_DIR "/test_pattern.toml");
    REQUIRE(schedule.passes.size() == 1);
    CHECK(schedule.passes[0] == PassKind::test_pattern);
}

TEST_CASE("a missing file is an error that names it") {
    try {
        (void)load_schedule("no/such/graph.toml");
        FAIL("expected an error");
    } catch (const GraphFileError& error) {
        CHECK(contains(error.what(), "no/such/graph.toml"));
    }
}
