#include "core/frame/schedule.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>

namespace serenity::frame {

namespace {

// Every kind, in declaration order: the list pass_kind() reads names back
// through. The switches below cover every kind (-Wswitch); this list is
// held to the enum by its length, so a kind added last and left out of it
// fails to compile (schedule.h).
constexpr std::array<PassKind, 5> kinds = {PassKind::test_pattern, PassKind::preview, PassKind::path,
                                           PassKind::display, PassKind::tone_map};
static_assert(kinds.size() == static_cast<std::size_t>(PassKind::tone_map) + 1,
              "every PassKind is in `kinds`, in declaration order");
static_assert(kinds.back() == PassKind::tone_map, "the last PassKind is last in `kinds`");

[[noreturn]] void no_case(const char* where) {
    // After a switch over every PassKind: reached only by a value no
    // enumerator names, which is refused rather than answered (P.6).
    throw std::logic_error(std::string(where) + ": a pass kind that is no PassKind");
}

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
    no_case("name");
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
    no_case("needs_scene");
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
    no_case("accumulates");
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
    no_case("writes_radiance");
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
    no_case("reads_radiance");
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
    no_case("writes_target");
}

namespace {

// The rules of schedule.h, invalid(), one function each (F.3), in the
// order they are checked: none if `passes` keeps the rule, else why not.

// At most one pass accumulates.
std::optional<std::string> accumulation_broken(const std::vector<PassKind>& passes) {
    std::vector<PassKind> accumulating;
    std::ranges::copy_if(passes, std::back_inserter(accumulating), [](PassKind kind) { return accumulates(kind); });
    if (accumulating.size() > 1) {
        return "the frame graph has " + std::to_string(accumulating.size()) + " passes that accumulate (" +
               listed(accumulating) + "); one accumulated image holds one pass's history, so a graph may have at "
               "most one";
    }
    return std::nullopt;
}

// One pass writes the target, the last.
std::optional<std::string> presentation_broken(const std::vector<PassKind>& passes) {
    std::vector<PassKind> presenting;
    std::ranges::copy_if(passes, std::back_inserter(presenting), [](PassKind kind) { return writes_target(kind); });
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
    return std::nullopt;
}

// At most one pass writes radiance; every pass that reads it follows it,
// and one does.
std::optional<std::string> radiance_broken(const std::vector<PassKind>& passes) {
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
    return std::nullopt;
}

}  // namespace

std::optional<std::string> invalid(const Schedule& schedule) {
    const std::vector<PassKind>& passes = schedule.passes;
    if (passes.empty()) {
        return "the frame graph has no passes";
    }
    for (const auto broken : {accumulation_broken, presentation_broken, radiance_broken}) {
        if (std::optional<std::string> reason = broken(passes)) {
            return reason;
        }
    }

    // The tone map's settings, exactly with its pass.
    const bool tone_maps = std::ranges::find(passes, PassKind::tone_map) != passes.end();
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

std::optional<std::string> invalid(const passes::ToneMap& settings) {
    if (!std::isfinite(settings.exposure) || std::abs(settings.exposure) > passes::max_exposure) {
        const std::string limit = std::to_string(static_cast<int>(passes::max_exposure));
        return "the tone map's exposure, " + std::to_string(settings.exposure) + " stops, must be finite, within [-" +
               limit + ", " + limit + "]";
    }
    if (!std::isfinite(settings.bloom) || settings.bloom < 0.0f || settings.bloom >= 1.0f) {
        return "the tone map's bloom, " + std::to_string(settings.bloom) + ", must be within [0, 1)";
    }
    return std::nullopt;
}

std::optional<PassKind> pass_kind(std::string_view text) {
    const auto found = std::ranges::find_if(kinds, [&](PassKind kind) { return name(kind) == text; });
    return found == kinds.end() ? std::nullopt : std::optional<PassKind>{*found};
}

std::span<const PassKind> all_pass_kinds() {
    return kinds;
}

bool accumulates(const Schedule& schedule) {
    return std::ranges::any_of(schedule.passes, [](PassKind kind) { return accumulates(kind); });
}

}  // namespace serenity::frame
