// Pass kinds and their names in scene files: one table, read both ways.

#include <limits>
#include <set>
#include <string>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"

using serenity::frame::all_pass_kinds;
using serenity::frame::name;
using serenity::frame::pass_kind;

TEST_CASE("every pass kind has a distinct name that reads back as itself") {
    std::set<std::string> names;
    for (auto kind : all_pass_kinds()) {
        const auto text = name(kind);
        CHECK_FALSE(text.empty());
        CHECK(names.insert(std::string(text)).second);
        CHECK(pass_kind(text) == kind);
    }
}

TEST_CASE("an unknown name is no pass kind") {
    CHECK_FALSE(pass_kind("test-pattern").has_value());
    CHECK_FALSE(pass_kind("").has_value());
}

TEST_CASE("which pass kinds read the scene") {
    CHECK_FALSE(serenity::frame::needs_scene(serenity::frame::PassKind::test_pattern));
    CHECK(serenity::frame::needs_scene(serenity::frame::PassKind::preview));
    CHECK(pass_kind("preview") == serenity::frame::PassKind::preview);
}

TEST_CASE("the path pass reads the scene and accumulates; the others do not accumulate") {
    using namespace serenity::frame;
    CHECK(needs_scene(PassKind::path));
    CHECK(accumulates(PassKind::path));
    CHECK_FALSE(accumulates(PassKind::preview));
    CHECK_FALSE(accumulates(PassKind::test_pattern));
    CHECK(pass_kind("path") == PassKind::path);
}

namespace {

using namespace serenity::frame;

constexpr ToneMap look{0.0f, 0.04f, {0.0f, 0.0f}};

std::string why(const Schedule& schedule) {
    return invalid(schedule).value_or("");
}

bool says(const Schedule& schedule, const char* part) {
    const std::string reason = why(schedule);
    INFO(reason);
    return reason.find(part) != std::string::npos;
}

}  // namespace

TEST_CASE("which pass kinds write and read which of the frame's images") {
    CHECK(writes_radiance(PassKind::preview));
    CHECK(writes_radiance(PassKind::path));
    CHECK(reads_radiance(PassKind::display));
    CHECK(reads_radiance(PassKind::tone_map));
    CHECK(writes_target(PassKind::display));
    CHECK(writes_target(PassKind::tone_map));
    CHECK(writes_target(PassKind::test_pattern));
    CHECK_FALSE(reads_radiance(PassKind::test_pattern));
    CHECK_FALSE(writes_target(PassKind::path));
    CHECK_FALSE(needs_scene(PassKind::display));
    CHECK_FALSE(needs_scene(PassKind::tone_map));
    CHECK_FALSE(accumulates(PassKind::tone_map));
}

TEST_CASE("which schedules are valid is the core's") {
    CHECK(why(Schedule{{PassKind::path, PassKind::display}, std::nullopt}).empty());
    CHECK(why(Schedule{{PassKind::path, PassKind::tone_map}, look}).empty());
    CHECK(why(Schedule{{PassKind::preview, PassKind::display}, std::nullopt}).empty());
    CHECK(why(Schedule{{PassKind::test_pattern}, std::nullopt}).empty());
    CHECK(says(Schedule{}, "no passes"));
    CHECK(says(Schedule{{PassKind::path, PassKind::path, PassKind::display}, std::nullopt}, "at most one"));
}

TEST_CASE("one pass writes the image shown, and it is the last") {
    CHECK(says(Schedule{{PassKind::path}, std::nullopt}, "no pass in the frame graph writes the image shown"));
    CHECK(says(Schedule{{PassKind::test_pattern, PassKind::test_pattern}, std::nullopt},
               "2 passes that write the image shown (test_pattern, test_pattern)"));
    CHECK(says(Schedule{{PassKind::test_pattern, PassKind::path}, std::nullopt}, "last pass, path"));
}

TEST_CASE("light is computed once, before it is shown, and is shown") {
    CHECK(says(Schedule{{PassKind::display}, std::nullopt}, "no pass before it computes light"));
    CHECK(says(Schedule{{PassKind::preview, PassKind::path, PassKind::display}, std::nullopt},
               "two passes that compute light (preview, path)"));
    CHECK(says(Schedule{{PassKind::path, PassKind::test_pattern}, std::nullopt},
               "path computes light and no pass after it shows it"));
}

TEST_CASE("tone-map settings come exactly with the pass, in range") {
    CHECK(says(Schedule{{PassKind::path, PassKind::tone_map}, std::nullopt}, "no tone-map settings"));
    CHECK(says(Schedule{{PassKind::path, PassKind::display}, look}, "no tone_map pass"));
    const auto with = [](float exposure, float bloom) {
        return Schedule{{PassKind::path, PassKind::tone_map}, ToneMap{exposure, bloom, {0.0f, 0.0f}}};
    };
    CHECK(why(with(-10.0f, 0.0f)).empty());
    CHECK(why(with(10.0f, 0.999f)).empty());
    CHECK(says(with(10.5f, 0.0f), "exposure"));
    CHECK(says(with(std::numeric_limits<float>::quiet_NaN(), 0.0f), "exposure"));
    CHECK(says(with(0.0f, 1.0f), "bloom"));
    CHECK(says(with(0.0f, -0.01f), "bloom"));
    CHECK(says(with(0.0f, std::numeric_limits<float>::infinity()), "bloom"));
}
