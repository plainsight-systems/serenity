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
