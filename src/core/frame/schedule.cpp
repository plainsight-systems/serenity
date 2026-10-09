#include "core/frame/schedule.h"

#include <array>
#include <cmath>

namespace serenity::frame {

namespace {

constexpr std::array<PassKind, 5> kinds = {PassKind::test_pattern, PassKind::preview, PassKind::path,
                                           PassKind::display, PassKind::tone_map};

// "a, b, c": the names of `passes`.
std::string listed(const std::vector<PassKind>& passes) {
    std::string text;
    for (PassKind kind : passes) {
        text += (text.empty() ? "" : ", ") + std::string(name(kind));
    }
    return text;
}

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
    case PassKind::display:
        return "display";
    case PassKind::tone_map:
        return "tone_map";
    }
    return {};  // unreachable for a valid PassKind
}

bool needs_scene(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::display:
    case PassKind::tone_map:
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
    case PassKind::display:
    case PassKind::tone_map:
        return false;
    case PassKind::path:
        return true;
    }
    return false;  // unreachable for a valid PassKind
}

bool writes_radiance(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::display:
    case PassKind::tone_map:
        return false;
    case PassKind::preview:
    case PassKind::path:
        return true;
    }
    return false;  // unreachable for a valid PassKind
}

bool reads_radiance(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::preview:
    case PassKind::path:
        return false;
    case PassKind::display:
    case PassKind::tone_map:
        return true;
    }
    return false;  // unreachable for a valid PassKind
}

bool writes_target(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::preview:
    case PassKind::path:
        return false;
    case PassKind::test_pattern:
    case PassKind::display:
    case PassKind::tone_map:
        return true;
    }
    return false;  // unreachable for a valid PassKind
}

std::optional<std::string> invalid(const Schedule& schedule) {
    const std::vector<PassKind>& passes = schedule.passes;
    if (passes.empty()) {
        return "the frame graph has no passes";
    }

    // At most one pass accumulates.
    std::vector<PassKind> accumulating;
    for (PassKind kind : passes) {
        if (accumulates(kind)) {
            accumulating.push_back(kind);
        }
    }
    if (accumulating.size() > 1) {
        return "the frame graph has " + std::to_string(accumulating.size()) + " passes that accumulate (" +
               listed(accumulating) + "); one accumulated image holds one pass's history, so a graph may have at "
               "most one";
    }

    // One pass writes the target, the last.
    std::vector<PassKind> presenting;
    for (PassKind kind : passes) {
        if (writes_target(kind)) {
            presenting.push_back(kind);
        }
    }
    if (presenting.empty()) {
        return "no pass in the frame graph writes the image shown (" + listed(passes) +
               "); end it with display or tone_map";
    }
    if (presenting.size() > 1) {
        return "the frame graph has " + std::to_string(presenting.size()) + " passes that write the image shown (" +
               listed(presenting) + "); a frame shows one image, so a graph may have one";
    }
    if (!writes_target(passes.back())) {
        return "the frame graph's last pass, " + std::string(name(passes.back())) +
               ", does not write the image shown; " + std::string(name(presenting.front())) +
               ", which does, must be last";
    }

    // At most one pass writes radiance; every pass that reads it follows it,
    // and one does.
    std::optional<PassKind> writer;
    bool shown = false;
    for (PassKind kind : passes) {
        if (reads_radiance(kind)) {
            if (!writer) {
                return std::string(name(kind)) +
                       " shows the light a pass before it computes, and no pass before it computes light";
            }
            shown = true;
        }
        if (writes_radiance(kind)) {
            if (writer) {
                return "the frame graph has two passes that compute light (" + std::string(name(*writer)) + ", " +
                       std::string(name(kind)) + "); a frame has one radiance image, so a graph may have one";
            }
            writer = kind;
        }
    }
    if (writer && !shown) {
        return std::string(name(*writer)) +
               " computes light and no pass after it shows it; end the frame graph with display or tone_map";
    }

    // The tone map's settings, exactly with its pass.
    bool tone_maps = false;
    for (PassKind kind : passes) {
        tone_maps = tone_maps || kind == PassKind::tone_map;
    }
    if (tone_maps && !schedule.tone_map) {
        return "the frame graph has a tone_map pass and no tone-map settings";
    }
    if (!tone_maps && schedule.tone_map) {
        return "the frame graph has tone-map settings and no tone_map pass";
    }
    if (schedule.tone_map) {
        return invalid(*schedule.tone_map);
    }
    return std::nullopt;
}

std::optional<std::string> invalid(const ToneMap& settings) {
    if (!std::isfinite(settings.exposure) || std::abs(settings.exposure) > max_exposure) {
        const std::string limit = std::to_string(static_cast<int>(max_exposure));
        return "the tone map's exposure, " + std::to_string(settings.exposure) + " stops, must be finite, within [-" +
               limit + ", " + limit + "]";
    }
    if (!std::isfinite(settings.bloom) || settings.bloom < 0.0f || settings.bloom >= 1.0f) {
        return "the tone map's bloom, " + std::to_string(settings.bloom) + ", must be within [0, 1)";
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
