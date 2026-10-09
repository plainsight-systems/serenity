#include "core/frame/schedule.h"

#include <array>

namespace serenity::frame {

namespace {

constexpr std::array<PassKind, 2> kinds = {PassKind::test_pattern, PassKind::preview};

}  // namespace

std::string_view name(PassKind kind) {
    // No default: a kind without a name fails to compile (-Wswitch, -Werror).
    switch (kind) {
    case PassKind::test_pattern:
        return "test_pattern";
    case PassKind::preview:
        return "preview";
    }
    return {};  // unreachable for a valid PassKind
}

bool needs_scene(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
        return false;
    case PassKind::preview:
        return true;
    }
    return true;  // unreachable for a valid PassKind
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

}  // namespace serenity::frame
