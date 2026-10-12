#include "core/frame/schedule.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace serenity::frame {

namespace {

// Every kind, in declaration order: the list pass_kind() reads names back
// through. The switches below cover every kind (-Wswitch); this list is
// held to the enum's final kind, PassKind::last, so a kind left out of it
// fails to compile (schedule.h).
constexpr std::array<PassKind, 14> kinds = {
    PassKind::test_pattern,      PassKind::preview,           PassKind::path,
    PassKind::display,           PassKind::tone_map,          PassKind::restir_initial,
    PassKind::restir_temporal,   PassKind::restir_spatial,    PassKind::restir_shade,
    PassKind::restir_pt_initial, PassKind::restir_pt_temporal, PassKind::restir_pt_spatial,
    PassKind::restir_pt_shade,   PassKind::restir_pt_duplication};
static_assert(kinds.size() == static_cast<std::size_t>(PassKind::last) + 1,
              "every PassKind is in `kinds`, in declaration order");
static_assert(kinds.back() == PassKind::last, "the last PassKind is last in `kinds`");

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
    case PassKind::restir_initial:
        return "restir_initial";
    case PassKind::restir_temporal:
        return "restir_temporal";
    case PassKind::restir_spatial:
        return "restir_spatial";
    case PassKind::restir_shade:
        return "restir_shade";
    case PassKind::restir_pt_initial:
        return "restir_pt_initial";
    case PassKind::restir_pt_temporal:
        return "restir_pt_temporal";
    case PassKind::restir_pt_spatial:
        return "restir_pt_spatial";
    case PassKind::restir_pt_shade:
        return "restir_pt_shade";
    case PassKind::restir_pt_duplication:
        return "restir_pt_duplication";
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
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_shade:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
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
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_duplication:
        return false;
    case PassKind::path:
    case PassKind::restir_shade:     // folds its pixels into the accumulated image, as path does
    case PassKind::restir_pt_shade:  // likewise
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
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_duplication:
        return false;
    case PassKind::preview:
    case PassKind::path:
    case PassKind::restir_shade:
    case PassKind::restir_pt_shade:
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
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_shade:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
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
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_shade:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
        return false;
    case PassKind::test_pattern:
    case PassKind::display:
    case PassKind::tone_map:
        return true;
    }
    no_case("writes_target");
}

bool is_restir(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::preview:
    case PassKind::path:
    case PassKind::display:
    case PassKind::tone_map:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
        return false;
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_shade:
        return true;
    }
    no_case("is_restir");
}

bool is_restir_pt(PassKind kind) {
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::preview:
    case PassKind::path:
    case PassKind::display:
    case PassKind::tone_map:
    case PassKind::restir_initial:
    case PassKind::restir_temporal:
    case PassKind::restir_spatial:
    case PassKind::restir_shade:
        return false;
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
        return true;
    }
    no_case("is_restir_pt");
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

// ReSTIR DI's passes in their order, each at most once, restir_initial
// and restir_shade both present if any is (schedule.h). Nothing else can
// stand between them: every other kind that is not the presenting pass
// writes radiance or accumulates, which the rules before this one refuse
// beside the shade.
std::optional<std::string> restir_broken(const std::vector<PassKind>& passes) {
    constexpr std::array<PassKind, 4> order = {PassKind::restir_initial, PassKind::restir_temporal,
                                               PassKind::restir_spatial, PassKind::restir_shade};
    std::vector<PassKind> restir;
    std::ranges::copy_if(passes, std::back_inserter(restir), [](PassKind kind) { return is_restir(kind); });
    if (restir.empty()) {
        return std::nullopt;
    }
    for (PassKind kind : order) {
        if (std::ranges::count(restir, kind) > 1) {
            return "the frame graph has " + std::string(name(kind)) +
                   " more than once; ReSTIR DI runs each of its passes at most once";
        }
    }
    const auto has = [&](PassKind kind) { return std::ranges::find(restir, kind) != restir.end(); };
    if (!has(PassKind::restir_initial) || !has(PassKind::restir_shade)) {
        return "the frame graph has ReSTIR DI's passes (" + listed(restir) +
               ") without both restir_initial and restir_shade, which every ReSTIR DI graph begins and ends with";
    }
    const auto rank = [&](PassKind kind) { return std::ranges::find(order, kind) - order.begin(); };
    if (!std::ranges::is_sorted(restir, {}, rank)) {
        return "ReSTIR DI's passes run in the order restir_initial, restir_temporal, restir_spatial, "
               "restir_shade; the frame graph has " +
               listed(restir);
    }
    return std::nullopt;
}

// The pieces of the ReSTIR settings' refusals (invalid(RestirDi) below),
// each worded once.

// "an integer in [low, high]".
// ReSTIR PT's order (schedule.h), never beside ReSTIR DI's passes.
std::optional<std::string> restir_pt_broken(const std::vector<PassKind>& passes) {
    std::vector<PassKind> pt;
    std::ranges::copy_if(passes, std::back_inserter(pt), [](PassKind kind) { return is_restir_pt(kind); });
    if (pt.empty()) {
        return std::nullopt;
    }
    if (std::ranges::any_of(passes, [](PassKind kind) { return is_restir(kind); })) {
        return "the frame graph has ReSTIR DI's and ReSTIR PT's passes; ReSTIR PT's reservoir holds the direct light "
               "ReSTIR DI's would, so a graph runs one family";
    }
    constexpr std::array<PassKind, 5> order = {PassKind::restir_pt_initial, PassKind::restir_pt_temporal,
                                               PassKind::restir_pt_spatial, PassKind::restir_pt_shade,
                                               PassKind::restir_pt_duplication};
    for (PassKind kind : order) {
        if (std::ranges::count(pt, kind) > 1) {
            return "the frame graph has " + std::string(name(kind)) +
                   " more than once; ReSTIR PT runs each of its passes at most once";
        }
    }
    const auto has = [&](PassKind kind) { return std::ranges::find(pt, kind) != pt.end(); };
    if (!has(PassKind::restir_pt_initial) || !has(PassKind::restir_pt_shade)) {
        return "the frame graph has ReSTIR PT's passes (" + listed(pt) +
               ") without both restir_pt_initial and restir_pt_shade, which every ReSTIR PT graph begins and ends "
               "with";
    }
    const auto rank = [&](PassKind kind) { return std::ranges::find(order, kind) - order.begin(); };
    if (!std::ranges::is_sorted(pt, {}, rank)) {
        return "ReSTIR PT's passes run in the order restir_pt_initial, restir_pt_temporal, restir_pt_spatial, "
               "restir_pt_shade, restir_pt_duplication; the frame graph has " +
               listed(pt);
    }
    if (has(PassKind::restir_pt_duplication) && !has(PassKind::restir_pt_temporal)) {
        return "the frame graph has restir_pt_duplication without restir_pt_temporal; the duplication map lowers "
               "the temporal reuse's cap, and serves nothing else";
    }
    return std::nullopt;
}

std::string integer_in(std::uint32_t low, std::uint32_t high) {
    return "an integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
}

// "ReSTIR DI's <key>, <value>, ": the start of every refusal.
std::string setting(std::string_view key, const std::string& value) {
    return "ReSTIR DI's " + std::string(key) + ", " + value + ", ";
}

// A setting the schedule's passes do not read must be its zero
// (core/contracts/reuse_settings.h); `readers` says which passes read it.
std::string unread(std::string_view key, const std::string& value, std::string_view readers) {
    return setting(key, value) + "is read only " + std::string(readers) +
           ", and must be its zero (left out of the graph file) otherwise";
}

// A float as the graph file writes it: 0.1, not 0.100000.
std::string text(float value) {
    std::ostringstream s;
    s << value;
    return s.str();
}

// An enumerator's value, which may be one no enumerator names.
template <typename Enum>
std::string text(Enum value) {
    return std::to_string(static_cast<std::uint32_t>(value));
}

bool finite_within(float value, float low, float high) {
    return std::isfinite(value) && value >= low && value <= high;
}

// invalid(RestirDi)'s rules, by the passes that read each group of
// settings (F.3): the candidates' (always read), the temporal merge's, the
// spatial merge's, either merge's, and the shade's final-visibility reuse.
std::optional<std::string> candidates_broken(const passes::RestirDi& settings) {
    if (settings.candidates < 1u || settings.candidates > contracts::max_candidates) {
        return setting("candidates", std::to_string(settings.candidates)) + "must be " +
               integer_in(1u, contracts::max_candidates);
    }
    if (settings.bsdf_candidates > contracts::max_bsdf_candidates) {
        return setting("bsdf_candidates", std::to_string(settings.bsdf_candidates)) + "must be " +
               integer_in(0u, contracts::max_bsdf_candidates);
    }
    return std::nullopt;
}

std::optional<std::string> temporal_broken(const passes::RestirDi& settings, bool temporal) {
    constexpr std::string_view readers = "with restir_temporal";
    if (temporal && (settings.history < 1u || settings.history > contracts::max_history)) {
        return setting("history", std::to_string(settings.history)) + "must be " + integer_in(1u, contracts::max_history);
    }
    if (!temporal && settings.history != 0u) {
        return unread("history", std::to_string(settings.history), readers);
    }
    if (settings.permutation != contracts::Permutation::off && settings.permutation != contracts::Permutation::on) {
        return setting("permutation", text(settings.permutation)) + "is neither off nor on";
    }
    if (!temporal && settings.permutation != contracts::Permutation::off) {
        return unread("permutation", "on", readers);
    }
    return std::nullopt;
}

std::optional<std::string> spatial_broken(const passes::RestirDi& settings, bool spatial) {
    constexpr std::string_view readers = "with restir_spatial";
    if (spatial && (settings.neighbors < 1u || settings.neighbors > contracts::max_neighbors)) {
        return setting("neighbors", std::to_string(settings.neighbors)) + "must be " +
               integer_in(1u, contracts::max_neighbors);
    }
    if (!spatial && settings.neighbors != 0u) {
        return unread("neighbors", std::to_string(settings.neighbors), readers);
    }
    if (spatial && !finite_within(settings.radius, contracts::min_radius, contracts::max_radius)) {
        return setting("radius", text(settings.radius)) + "must be finite, within [" + text(contracts::min_radius) +
               ", " + text(contracts::max_radius) + "] pixels";
    }
    if (!spatial && settings.radius != 0.0f) {
        return unread("radius", text(settings.radius), readers);
    }
    return std::nullopt;
}

std::optional<std::string> merges_broken(const passes::RestirDi& settings, bool merges) {
    constexpr std::string_view readers = "with restir_temporal or restir_spatial";
    const float depth = settings.depth_threshold;
    if (merges && !(std::isfinite(depth) && depth > 0.0f && depth <= 1.0f)) {
        return setting("depth_threshold", text(depth)) + "must be within (0, 1]";
    }
    if (!merges && depth != 0.0f) {
        return unread("depth_threshold", text(depth), readers);
    }
    if (merges && !finite_within(settings.normal_threshold, -1.0f, 1.0f)) {
        return setting("normal_threshold", text(settings.normal_threshold)) + "must be a cosine, within [-1, 1]";
    }
    if (!merges && settings.normal_threshold != 0.0f) {
        return unread("normal_threshold", text(settings.normal_threshold), readers);
    }
    if (settings.bias_correction != contracts::BiasCorrection::basic &&
        settings.bias_correction != contracts::BiasCorrection::raytraced) {
        return setting("bias_correction", text(settings.bias_correction)) + "is neither basic nor raytraced";
    }
    if (!merges && settings.bias_correction != contracts::BiasCorrection::basic) {
        return unread("bias_correction", "raytraced", readers);
    }
    return std::nullopt;
}

std::optional<std::string> visibility_broken(const passes::RestirDi& settings, bool merges) {
    const bool basic = merges && settings.bias_correction == contracts::BiasCorrection::basic;
    if (settings.visibility_reuse != contracts::VisibilityReuse::off &&
        settings.visibility_reuse != contracts::VisibilityReuse::on) {
        return setting("visibility_reuse", text(settings.visibility_reuse)) + "is neither off nor on";
    }
    if (!basic && settings.visibility_reuse != contracts::VisibilityReuse::off) {
        return unread("visibility_reuse", "on", "in basic mode, with restir_temporal or restir_spatial");
    }
    const bool reused = basic && settings.visibility_reuse == contracts::VisibilityReuse::on;
    constexpr std::string_view readers = "where visibility_reuse is on";
    if (reused && settings.visibility_age > contracts::max_visibility_age) {
        return setting("visibility_age", std::to_string(settings.visibility_age)) + "must be " +
               integer_in(0u, contracts::max_visibility_age);
    }
    if (!reused && settings.visibility_age != 0u) {
        return unread("visibility_age", std::to_string(settings.visibility_age), readers);
    }
    if (reused && !finite_within(settings.visibility_distance, 0.0f, contracts::max_visibility_distance)) {
        return setting("visibility_distance", text(settings.visibility_distance)) + "must be finite, within [0, " +
               text(contracts::max_visibility_distance) + "] pixels";
    }
    if (!reused && settings.visibility_distance != 0.0f) {
        return unread("visibility_distance", text(settings.visibility_distance), readers);
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> invalid(const passes::RestirDi& settings, std::span<const PassKind> passes) {
    const auto has = [&](PassKind kind) { return std::ranges::find(passes, kind) != passes.end(); };
    const bool temporal = has(PassKind::restir_temporal);
    const bool spatial = has(PassKind::restir_spatial);
    if (std::optional<std::string> reason = candidates_broken(settings)) {
        return reason;
    }
    if (std::optional<std::string> reason = temporal_broken(settings, temporal)) {
        return reason;
    }
    if (std::optional<std::string> reason = spatial_broken(settings, spatial)) {
        return reason;
    }
    if (std::optional<std::string> reason = merges_broken(settings, temporal || spatial)) {
        return reason;
    }
    return visibility_broken(settings, temporal || spatial);
}

std::optional<std::string> invalid(const Schedule& schedule) {
    const std::vector<PassKind>& passes = schedule.passes;
    if (passes.empty()) {
        return "the frame graph has no passes";
    }
    for (const auto broken :
         {accumulation_broken, presentation_broken, radiance_broken, restir_pt_broken, restir_broken}) {
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
        if (std::optional<std::string> reason = invalid(*schedule.tone_map)) {
            return reason;
        }
    }

    // ReSTIR DI's settings, exactly with its passes.
    const bool restir = std::ranges::any_of(passes, [](PassKind kind) { return is_restir(kind); });
    if (restir && !schedule.restir_di) {
        return "the frame graph has ReSTIR DI's passes and no [restir_di] settings";
    }
    if (!restir && schedule.restir_di) {
        return "the frame graph has [restir_di] settings and no ReSTIR DI pass";
    }
    if (schedule.restir_di) {
        if (std::optional<std::string> reason = invalid(*schedule.restir_di, passes)) {
            return reason;
        }
    }

    const bool restir_pt = std::ranges::any_of(passes, [](PassKind kind) { return is_restir_pt(kind); });
    if (schedule.mnee && !restir_pt && std::ranges::find(passes, PassKind::path) == passes.end()) {
        return "the frame graph sets mnee and has no path or ReSTIR PT pass to read it";
    }
    if (restir_pt && !schedule.restir_pt) {
        return "the frame graph has ReSTIR PT's passes and no [restir_pt] settings";
    }
    if (!restir_pt && schedule.restir_pt) {
        return "the frame graph has [restir_pt] settings and no ReSTIR PT pass";
    }
    if (schedule.restir_pt) {
        return invalid(*schedule.restir_pt, passes);
    }
    return std::nullopt;
}

std::optional<std::string> invalid(const passes::RestirPt& settings, std::span<const PassKind> passes) {
    const bool spatial = std::ranges::find(passes, PassKind::restir_pt_spatial) != passes.end();
    const bool temporal = std::ranges::find(passes, PassKind::restir_pt_temporal) != passes.end();
    const bool duplication = std::ranges::find(passes, PassKind::restir_pt_duplication) != passes.end();
    const auto setting = [](std::string_view key, const std::string& value) {
        return "ReSTIR PT's " + std::string(key) + ", " + value + ", ";
    };
    if (settings.nee_candidates < 1u || settings.nee_candidates > contracts::max_nee_candidates) {
        return setting("nee_candidates", std::to_string(settings.nee_candidates)) + "must be " +
               integer_in(1u, contracts::max_nee_candidates);
    }
    if (settings.max_surfaces < 1u || settings.max_surfaces > contracts::max_path_surfaces) {
        return setting("max_surfaces", std::to_string(settings.max_surfaces)) + "must be " +
               integer_in(1u, contracts::max_path_surfaces);
    }
    if (!(std::isfinite(settings.footprint) && settings.footprint > 0.0f &&
          settings.footprint <= contracts::max_footprint)) {
        return setting("footprint", text(settings.footprint)) + "must be finite, within (0, " +
               text(contracts::max_footprint) + "]";
    }
    if (!finite_within(settings.roughness, 0.0f, 1.0f)) {
        return setting("roughness", text(settings.roughness)) + "must be finite, within [0, 1]";
    }
    if (spatial && (settings.neighbors < 1u || settings.neighbors > contracts::max_path_neighbors)) {
        return setting("neighbors", std::to_string(settings.neighbors)) + "must be " +
               integer_in(1u, contracts::max_path_neighbors);
    }
    if (!spatial && settings.neighbors != 0u) {
        return setting("neighbors", std::to_string(settings.neighbors)) +
               "is read only with restir_pt_spatial, and must be its zero (left out of the graph file) otherwise";
    }
    if (spatial && !finite_within(settings.sigma, contracts::min_pairing_sigma, contracts::max_pairing_sigma)) {
        return setting("sigma", text(settings.sigma)) + "must be finite, within [" +
               text(contracts::min_pairing_sigma) + ", " + text(contracts::max_pairing_sigma) + "] pixels";
    }
    if (!spatial && settings.sigma != 0.0f) {
        return setting("sigma", text(settings.sigma)) +
               "is read only with restir_pt_spatial, and must be its zero (left out of the graph file) otherwise";
    }
    const auto unread_by = [&](std::string_view key, float value, std::string_view pass) {
        return setting(key, text(value)) + "is read only with " + std::string(pass) +
               ", and must be its zero (left out of the graph file) otherwise";
    };
    if (temporal && !finite_within(settings.confidence_cap, 1.0f, contracts::max_confidence_cap)) {
        return setting("confidence_cap", text(settings.confidence_cap)) + "must be finite, within [1, " +
               text(contracts::max_confidence_cap) + "]";
    }
    if (!temporal && settings.confidence_cap != 0.0f) {
        return unread_by("confidence_cap", settings.confidence_cap, "restir_pt_temporal");
    }
    // Bounded by the cap where temporal reuse reads one; a map without it is
    // the passes' rule's to refuse (restir_pt_broken), by its own words.
    if (duplication && !finite_within(settings.cap_floor, 0.0f,
                                      temporal ? settings.confidence_cap : contracts::max_confidence_cap)) {
        return setting("cap_floor", text(settings.cap_floor)) + "must be finite, within [0, confidence_cap]";
    }
    if (!duplication && settings.cap_floor != 0.0f) {
        return unread_by("cap_floor", settings.cap_floor, "restir_pt_duplication");
    }
    if (duplication && !(std::isfinite(settings.cap_exponent) && settings.cap_exponent > 0.0f &&
                         settings.cap_exponent <= 1.0f)) {
        return setting("cap_exponent", text(settings.cap_exponent)) + "must be finite, within (0, 1]";
    }
    if (!duplication && settings.cap_exponent != 0.0f) {
        return unread_by("cap_exponent", settings.cap_exponent, "restir_pt_duplication");
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

bool reuses_history(const Schedule& schedule) {
    return std::ranges::any_of(schedule.passes, [](PassKind kind) {
        return kind == PassKind::restir_temporal || kind == PassKind::restir_pt_temporal;
    });
}

}  // namespace serenity::frame
