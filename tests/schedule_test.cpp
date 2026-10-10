// Pass kinds and their names in graph files: one table, read both ways; what
// each kind reads and writes; and which schedules the core accepts.

#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"
#include "support/text.h"

using namespace serenity::frame;
using serenity::passes::ToneMap;

namespace {

constexpr ToneMap look{0.0f, 0.04f, {0.0f, 0.0f}};

// Why the core refuses `schedule`, or "" if it accepts it.
std::string refusal(const Schedule& schedule) {
    return invalid(schedule).value_or("");
}

// Whether the core refuses `schedule` for a reason that says `part`.
bool refused_for(const Schedule& schedule, std::string_view part) {
    const std::string reason = refusal(schedule);
    INFO(reason);
    return serenity::tests::contains(reason, part);
}

}  // namespace

TEST_CASE("every pass kind has a distinct name that reads back as itself") {
    std::set<std::string> names;
    for (const PassKind kind : all_pass_kinds()) {
        const std::string_view text = name(kind);
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
    CHECK_FALSE(needs_scene(PassKind::test_pattern));
    CHECK(needs_scene(PassKind::preview));
    CHECK(pass_kind("preview") == PassKind::preview);
}

TEST_CASE("the path pass reads the scene and accumulates; the others do not accumulate") {
    CHECK(needs_scene(PassKind::path));
    CHECK(accumulates(PassKind::path));
    CHECK_FALSE(accumulates(PassKind::preview));
    CHECK_FALSE(accumulates(PassKind::test_pattern));
    CHECK(pass_kind("path") == PassKind::path);
}

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
    CHECK(refusal(Schedule{{PassKind::path, PassKind::display}, std::nullopt}).empty());
    CHECK(refusal(Schedule{{PassKind::path, PassKind::tone_map}, look}).empty());
    CHECK(refusal(Schedule{{PassKind::preview, PassKind::display}, std::nullopt}).empty());
    CHECK(refusal(Schedule{{PassKind::test_pattern}, std::nullopt}).empty());
    CHECK(refused_for(Schedule{}, "no passes"));
    CHECK(refused_for(Schedule{{PassKind::path, PassKind::path, PassKind::display}, std::nullopt}, "at most one"));
}

TEST_CASE("one pass writes the image shown, and it is the last") {
    CHECK(refused_for(Schedule{{PassKind::path}, std::nullopt}, "no pass in the frame graph writes the image shown"));
    CHECK(refused_for(Schedule{{PassKind::test_pattern, PassKind::test_pattern}, std::nullopt},
                      "2 passes that write the image shown (test_pattern, test_pattern)"));
    CHECK(refused_for(Schedule{{PassKind::test_pattern, PassKind::path}, std::nullopt}, "last pass, path"));
}

TEST_CASE("light is computed once, before it is shown, and is shown") {
    CHECK(refused_for(Schedule{{PassKind::display}, std::nullopt}, "no pass before it computes light"));
    CHECK(refused_for(Schedule{{PassKind::preview, PassKind::path, PassKind::display}, std::nullopt},
                      "two passes that compute light (preview, path)"));
    CHECK(refused_for(Schedule{{PassKind::path, PassKind::test_pattern}, std::nullopt},
                      "path computes light and no pass after it shows it"));
}

TEST_CASE("tone-map settings come exactly with the pass, in range") {
    CHECK(refused_for(Schedule{{PassKind::path, PassKind::tone_map}, std::nullopt}, "no tone-map settings"));
    CHECK(refused_for(Schedule{{PassKind::path, PassKind::display}, look}, "no tone_map pass"));
    const auto with = [](float exposure, float bloom) {
        return Schedule{{PassKind::path, PassKind::tone_map}, ToneMap{exposure, bloom, {0.0f, 0.0f}}};
    };
    CHECK(refusal(with(-10.0f, 0.0f)).empty());
    CHECK(refusal(with(10.0f, 0.999f)).empty());
    CHECK(refused_for(with(10.5f, 0.0f), "exposure"));
    CHECK(refused_for(with(std::numeric_limits<float>::quiet_NaN(), 0.0f), "exposure"));
    CHECK(refused_for(with(0.0f, 1.0f), "bloom"));
    CHECK(refused_for(with(0.0f, -0.01f), "bloom"));
    CHECK(refused_for(with(0.0f, std::numeric_limits<float>::infinity()), "bloom"));
}

TEST_CASE("every kind reads back by its name, and a value no kind names is refused") {
    for (const PassKind kind : all_pass_kinds()) {
        CHECK(pass_kind(name(kind)) == kind);
    }
    const auto stray = static_cast<PassKind>(9);
    CHECK_THROWS_AS((void)name(stray), std::logic_error);
    CHECK_THROWS_AS((void)needs_scene(stray), std::logic_error);
    CHECK_THROWS_AS((void)accumulates(stray), std::logic_error);
    CHECK_THROWS_AS((void)writes_radiance(stray), std::logic_error);
    CHECK_THROWS_AS((void)reads_radiance(stray), std::logic_error);
    CHECK_THROWS_AS((void)writes_target(stray), std::logic_error);
}
