#include "core/frame/schedule.h"

#include <array>

namespace serenity::frame {

namespace {

constexpr std::array<PassKind, 3> kinds = {PassKind::test_pattern, PassKind::preview, PassKind::path};

}  // namespace

std::string_view name(PassKind kind) {
    // No default: a kind without a name fails to compile (-Wswitch, -Werror).
    switch (kind) {
    case PassKind::test_pattern:
        return "test_pattern";
    case PassKind::preview:
        return "preview";
    case PassKind::path:
        return "path";
    }
    return {};  // unreachable for a valid PassKind
}

bool needs_scene(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
        return false;
    case PassKind::preview:
    case PassKind::path:
        return true;
    }
    return true;  // unreachable for a valid PassKind
}

bool accumulates(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::preview:
        return false;
    case PassKind::path:
        return true;
    }
    return false;  // unreachable for a valid PassKind
}

std::optional<std::string> invalid(const Schedule& schedule) {
    if (schedule.passes.empty()) {
        return "the frame graph has no passes";
    }
    std::string accumulating;
    std::size_t count = 0;
    for (PassKind kind : schedule.passes) {
        if (accumulates(kind)) {
            accumulating += (count++ == 0 ? "" : ", ") + std::string(name(kind));
        }
    }
    if (count > 1) {
        return "the frame graph has " + std::to_string(count) + " passes that accumulate (" + accumulating +
               "); one accumulated image holds one pass's history, so a graph may have at most one";
    }
    return std::nullopt;
}

std::optional<PassKind> pass_kind(std::string_view text) {
    for (PassKind kind : kinds) {
        if (name(kind) == text) {
            return kind;
        }
    }
    return std::nullopt;
}

std::span<const PassKind> all_pass_kinds() {
    return kinds;
}

bool accumulates(const Schedule& schedule) {
    for (PassKind kind : schedule.passes) {
        if (accumulates(kind)) {
            return true;
        }
    }
    return false;
}

}  // namespace serenity::frame
