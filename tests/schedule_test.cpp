// Pass kinds and their names in scene files: one table, read both ways.

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

TEST_CASE("which schedules are valid is the core's") {
    using namespace serenity::frame;
    CHECK_FALSE(invalid(Schedule{{PassKind::path}}).has_value());
    CHECK_FALSE(invalid(Schedule{{PassKind::test_pattern, PassKind::path}}).has_value());
    REQUIRE(invalid(Schedule{}).has_value());
    CHECK(invalid(Schedule{})->find("no passes") != std::string::npos);
    REQUIRE(invalid(Schedule{{PassKind::path, PassKind::path}}).has_value());
    CHECK(invalid(Schedule{{PassKind::path, PassKind::path}})->find("at most one") != std::string::npos);
}
