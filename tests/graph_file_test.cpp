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
    const auto schedule = parse_schedule("passes = [\"preview\", \"display\"]\n", "graph.toml");
    REQUIRE(schedule.passes.size() == 2);
    CHECK(schedule.passes[0] == PassKind::preview);
    CHECK(schedule.passes[1] == PassKind::display);
    CHECK_FALSE(schedule.tone_map.has_value());
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

TEST_CASE("a graph with two passes that accumulate is refused, at the core, naming them") {
    const std::string error = error_for("passes = [\"path\", \"path\", \"display\"]\n");
    CHECK(contains(error, "graph.toml:1:"));
    CHECK(contains(error, "2 passes that accumulate (path, path)"));
    CHECK(error_for("passes = [\"path\", \"display\"]\n").empty());
}

TEST_CASE("a graph that computes light and does not show it is refused, at the core") {
    CHECK(contains(error_for("passes = [\"path\", \"test_pattern\"]\n"),
                   "graph.toml:1: path computes light and no pass after it shows it"));
}

TEST_CASE("the tone map's settings are read with its pass") {
    const auto schedule = parse_schedule(
        "passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = -1.5\nbloom = 0.25\n", "graph.toml");
    REQUIRE(schedule.tone_map.has_value());
    CHECK(schedule.tone_map->exposure == -1.5f);
    CHECK(schedule.tone_map->bloom == 0.25f);
    // A whole number is a number.
    CHECK(parse_schedule("passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = 1\nbloom = 0\n", "g")
              .tone_map->exposure == 1.0f);
}

TEST_CASE("every mistake in the tone map's settings is an error at its line") {
    const char* passes = "passes = [\"path\", \"tone_map\"]\n";
    const auto with = [&](const char* rest) { return error_for((std::string(passes) + rest).c_str()); };
    CHECK(contains(with(""), "graph.toml:1: the frame graph has a tone_map pass and no tone-map settings"));
    CHECK(contains(with("[tone_map]\nexposure = 0.0\n"), "graph.toml:2: [tone_map] has no 'bloom'"));
    CHECK(contains(with("[tone_map]\nexposure = \"0\"\nbloom = 0.0\n"), "graph.toml:3: 'exposure' must be a number"));
    CHECK(contains(with("[tone_map]\nexposure = 0.0\nbloom = 0.0\nglare = 1\n"),
                   "graph.toml:5: unknown key 'glare' in [tone_map]"));
    CHECK(contains(with("[tone_map]\nexposure = 11.0\nbloom = 0.0\n"), "graph.toml:2: the tone map's exposure"));
    CHECK(contains(with("[tone_map]\nexposure = 0.0\nbloom = 1.0\n"), "graph.toml:2: the tone map's bloom"));
    CHECK(contains(with("tone_map = 1\n"), "'tone_map' must be a table"));
    CHECK(contains(error_for("passes = [\"path\", \"display\"]\n[tone_map]\nexposure = 0.0\nbloom = 0.0\n"),
                   "tone-map settings and no tone_map pass"));
}

TEST_CASE("the path tracer's graph in graphs/ reads") {
    const auto schedule = serenity::frame::load_schedule(SERENITY_GRAPHS_DIR "/path.toml");
    REQUIRE(schedule.passes.size() == 2);
    CHECK(schedule.passes[0] == serenity::frame::PassKind::path);
    CHECK(schedule.passes[1] == serenity::frame::PassKind::tone_map);
    REQUIRE(schedule.tone_map.has_value());
    CHECK(schedule.tone_map->exposure == 0.0f);
    CHECK(schedule.tone_map->bloom == 0.04f);
}

TEST_CASE("the preview's graph in graphs/ reads") {
    const auto schedule = serenity::frame::load_schedule(SERENITY_GRAPHS_DIR "/preview.toml");
    REQUIRE(schedule.passes.size() == 2);
    CHECK(schedule.passes[1] == serenity::frame::PassKind::display);
}
