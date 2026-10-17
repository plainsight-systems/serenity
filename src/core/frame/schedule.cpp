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

#include "core/contracts/checkerboard.h"
#include "core/contracts/denoise.h"

namespace serenity::frame {

namespace {

// Every kind, in declaration order: the list pass_kind() reads names back
// through. The switches below cover every kind (-Wswitch); this list is
// held to the enum's final kind, PassKind::last, so a kind left out of it
// fails to compile (schedule.h).
constexpr std::array<PassKind, 12> kinds = {
    PassKind::test_pattern,       PassKind::preview,           PassKind::path,
    PassKind::display,            PassKind::tone_map,          PassKind::restir_pt_initial,
    PassKind::restir_pt_temporal, PassKind::restir_pt_spatial, PassKind::restir_pt_shade,
    PassKind::restir_pt_duplication, PassKind::features,       PassKind::denoise};
static_assert(kinds.size() == static_cast<std::size_t>(PassKind::last) + 1,
              "every PassKind is in `kinds`, in declaration order");
static_assert(kinds.back() == PassKind::last, "the last PassKind is last in `kinds`");

[[noreturn]] void no_case(const char* where) {
    // After a switch over every PassKind: reached only by a value no
    // enumerator names, which is refused rather than answered (P.6).
    throw std::logic_error(std::string(where) + ": a pass kind that is no PassKind");
}

// "a, b, c": the names of `passes`.
std::string listed(std::span<const PassKind> passes) {
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
    case PassKind::features:
        return "features";
    case PassKind::denoise:
        return "denoise";
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
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
    case PassKind::features:
    case PassKind::denoise:  // traces the full-size albedo it remodulates by (metal/frame/renderer.h)
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
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_duplication:
    case PassKind::features:
    case PassKind::denoise:
        return false;
    case PassKind::path:
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
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_duplication:
    case PassKind::features:
    case PassKind::denoise:  // the frame's radiance at the output size, not a light pass's (schedule.h)
        return false;
    case PassKind::preview:
    case PassKind::path:
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
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
    case PassKind::features:
        return false;
    case PassKind::display:
    case PassKind::tone_map:
    case PassKind::denoise:
        return true;
    }
    no_case("reads_radiance");
}

bool writes_target(PassKind kind) {
    // No default, as for name().
    switch (kind) {
    case PassKind::preview:
    case PassKind::path:
    case PassKind::restir_pt_initial:
    case PassKind::restir_pt_temporal:
    case PassKind::restir_pt_spatial:
    case PassKind::restir_pt_shade:
    case PassKind::restir_pt_duplication:
    case PassKind::features:
    case PassKind::denoise:
        return false;
    case PassKind::test_pattern:
    case PassKind::display:
    case PassKind::tone_map:
        return true;
    }
    no_case("writes_target");
}

bool writes_features(PassKind kind) {
    return kind == PassKind::features;
}

bool is_restir_pt(PassKind kind) {
    switch (kind) {
    case PassKind::test_pattern:
    case PassKind::preview:
    case PassKind::path:
    case PassKind::display:
    case PassKind::tone_map:
    case PassKind::features:
    case PassKind::denoise:
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

// One pass writes the target, the last; or the graph is the features pass
// alone. The features pass at most once.
std::optional<std::string> presentation_broken(const std::vector<PassKind>& passes) {
    if (std::ranges::count(passes, PassKind::features) > 1) {
        return "the frame graph has features more than once; a frame has one set of feature images";
    }
    std::vector<PassKind> presenting;
    std::ranges::copy_if(passes, std::back_inserter(presenting), [](PassKind kind) { return writes_target(kind); });
    if (presenting.empty() && passes.size() == 1 && passes.front() == PassKind::features) {
        return std::nullopt;  // the features alone, at a size of their own (schedule.h)
    }
    if (presenting.empty()) {
        return "no pass in the frame graph writes the image shown (" + listed(passes) +
               "); end it with display or tone_map (only the features pass alone shows nothing)";
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

// A family of passes kept in its order (schedule.h): those of `passes`
// that `is_member` names each at most once, `first` and `last`, which
// every graph of the family begins and ends with, both present if any is,
// in `order`. `family`, as "ReSTIR PT", names it in the refusals.
std::optional<std::string> family_broken(const std::vector<PassKind>& passes, bool (*is_member)(PassKind),
                                         std::span<const PassKind> order, PassKind first, PassKind last,
                                         std::string_view family) {
    std::vector<PassKind> members;
    std::ranges::copy_if(passes, std::back_inserter(members), is_member);
    if (members.empty()) {
        return std::nullopt;
    }
    for (PassKind kind : order) {
        if (std::ranges::count(members, kind) > 1) {
            return "the frame graph has " + std::string(name(kind)) + " more than once; " + std::string(family) +
                   " runs each of its passes at most once";
        }
    }
    const auto has = [&](PassKind kind) { return std::ranges::find(members, kind) != members.end(); };
    if (!has(first) || !has(last)) {
        return "the frame graph has " + std::string(family) + "'s passes (" + listed(members) + ") without both " +
               std::string(name(first)) + " and " + std::string(name(last)) + ", which every " +
               std::string(family) + " graph begins and ends with";
    }
    const auto rank = [&](PassKind kind) { return std::ranges::find(order, kind) - order.begin(); };
    if (!std::ranges::is_sorted(members, {}, rank)) {
        return std::string(family) + "'s passes run in the order " + listed(order) + "; the frame graph has " +
               listed(members);
    }
    return std::nullopt;
}

// ReSTIR PT's passes in their order, each at most once, restir_pt_initial
// and restir_pt_shade both present if any is, and the duplication map only
// with the temporal reuse (schedule.h).
std::optional<std::string> restir_pt_broken(const std::vector<PassKind>& passes) {
    if (std::ranges::none_of(passes, is_restir_pt)) {
        return std::nullopt;
    }
    constexpr std::array<PassKind, 5> order = {PassKind::restir_pt_initial, PassKind::restir_pt_temporal,
                                               PassKind::restir_pt_spatial, PassKind::restir_pt_shade,
                                               PassKind::restir_pt_duplication};
    if (std::optional<std::string> reason = family_broken(passes, is_restir_pt, order, PassKind::restir_pt_initial,
                                                          PassKind::restir_pt_shade, "ReSTIR PT")) {
        return reason;
    }
    const auto has = [&](PassKind kind) { return std::ranges::find(passes, kind) != passes.end(); };
    if (has(PassKind::restir_pt_duplication) && !has(PassKind::restir_pt_temporal)) {
        return "the frame graph has restir_pt_duplication without restir_pt_temporal; the duplication map lowers "
               "the temporal reuse's cap, and serves nothing else";
    }
    return std::nullopt;
}

// The denoise pass at most once, after the features pass and ReSTIR PT's
// shade, and directly before the presenting pass, the last (schedule.h).
// That a light pass computes radiance before any pass reads it is
// radiance_broken()'s.
std::optional<std::string> denoise_broken(const std::vector<PassKind>& passes) {
    const auto at = std::ranges::find(passes, PassKind::denoise);
    if (at == passes.end()) {
        return std::nullopt;
    }
    if (std::ranges::count(passes, PassKind::denoise) > 1) {
        return "the frame graph has denoise more than once; a frame is denoised once";
    }
    const auto before = [&](auto is) { return std::any_of(passes.begin(), at, is); };
    if (!before([](PassKind kind) { return writes_features(kind); })) {
        return "the frame graph's denoise pass reads the feature images, and no features pass comes before it";
    }
    if (!before([](PassKind kind) { return kind == PassKind::restir_pt_shade; })) {
        return "the frame graph's denoise pass reads ReSTIR PT's light, traced checkerboarded, and no "
               "restir_pt_shade comes before it";
    }
    if (passes.end() - at != 2 || !writes_target(passes.back())) {
        return "the frame graph's denoise pass writes the radiance its presenting pass shows at the output size, "
               "so it must come directly before that pass, the last";
    }
    return std::nullopt;
}

// The pieces of ReSTIR PT's settings' refusals (invalid(RestirPt) below),
// each worded once.

// The family whose settings they are, as the refusals name it.
constexpr std::string_view pt = "ReSTIR PT's";

// "an integer in [low, high]".
std::string integer_in(std::uint32_t low, std::uint32_t high) {
    return "an integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
}

// "<family> <key>, <value>, ": the start of every refusal.
std::string setting(std::string_view family, std::string_view key, const std::string& value) {
    return std::string(family) + " " + std::string(key) + ", " + value + ", ";
}

// A setting the schedule's passes do not read must be its zero
// (core/contracts/path_reuse_settings.h); `readers` says
// which passes read it.
std::string unread(std::string_view family, std::string_view key, const std::string& value,
                   std::string_view readers) {
    return setting(family, key, value) + "is read only " + std::string(readers) +
           ", and must be its zero (left out of the graph file) otherwise";
}

// A float as the graph file writes it: 0.1, not 0.100000.
std::string text(float value) {
    std::ostringstream s;
    s << value;
    return s.str();
}

bool finite_within(float value, float low, float high) {
    return std::isfinite(value) && value >= low && value <= high;
}

// invalid(RestirPt)'s rules, by the passes that read each group of settings:
// the four every ReSTIR PT graph reads (the path tree's and its shifts'),
// the spatial reuse's, the temporal reuse's, and the duplication map's.
std::optional<std::string> pt_paths_broken(const passes::RestirPt& settings) {
    if (settings.nee_candidates < 1u || settings.nee_candidates > contracts::max_nee_candidates) {
        return setting(pt, "nee_candidates", std::to_string(settings.nee_candidates)) + "must be " +
               integer_in(1u, contracts::max_nee_candidates);
    }
    if (settings.max_surfaces < 1u || settings.max_surfaces > contracts::max_path_surfaces) {
        return setting(pt, "max_surfaces", std::to_string(settings.max_surfaces)) + "must be " +
               integer_in(1u, contracts::max_path_surfaces);
    }
    if (!(std::isfinite(settings.footprint) && settings.footprint > 0.0f &&
          settings.footprint <= contracts::max_footprint)) {
        return setting(pt, "footprint", text(settings.footprint)) + "must be finite, within (0, " +
               text(contracts::max_footprint) + "]";
    }
    if (!finite_within(settings.roughness, 0.0f, 1.0f)) {
        return setting(pt, "roughness", text(settings.roughness)) + "must be finite, within [0, 1]";
    }
    return std::nullopt;
}

std::optional<std::string> pt_spatial_broken(const passes::RestirPt& settings, bool spatial) {
    constexpr std::string_view readers = "with restir_pt_spatial";
    if (spatial && (settings.neighbors < 1u || settings.neighbors > contracts::max_path_neighbors)) {
        return setting(pt, "neighbors", std::to_string(settings.neighbors)) + "must be " +
               integer_in(1u, contracts::max_path_neighbors);
    }
    if (!spatial && settings.neighbors != 0u) {
        return unread(pt, "neighbors", std::to_string(settings.neighbors), readers);
    }
    if (spatial && !finite_within(settings.sigma, contracts::min_pairing_sigma, contracts::max_pairing_sigma)) {
        return setting(pt, "sigma", text(settings.sigma)) + "must be finite, within [" +
               text(contracts::min_pairing_sigma) + ", " + text(contracts::max_pairing_sigma) + "] pixels";
    }
    if (!spatial && settings.sigma != 0.0f) {
        return unread(pt, "sigma", text(settings.sigma), readers);
    }
    return std::nullopt;
}

std::optional<std::string> pt_temporal_broken(const passes::RestirPt& settings, bool temporal) {
    if (temporal && !finite_within(settings.confidence_cap, 1.0f, contracts::max_confidence_cap)) {
        return setting(pt, "confidence_cap", text(settings.confidence_cap)) + "must be finite, within [1, " +
               text(contracts::max_confidence_cap) + "]";
    }
    if (!temporal && settings.confidence_cap != 0.0f) {
        return unread(pt, "confidence_cap", text(settings.confidence_cap), "with restir_pt_temporal");
    }
    return std::nullopt;
}

std::optional<std::string> pt_duplication_broken(const passes::RestirPt& settings, bool temporal, bool duplication) {
    constexpr std::string_view readers = "with restir_pt_duplication";
    // Bounded by the cap where temporal reuse reads one; a map without it is
    // the passes' rule's to refuse (restir_pt_broken), by its own words.
    if (duplication && !finite_within(settings.cap_floor, 0.0f,
                                      temporal ? settings.confidence_cap : contracts::max_confidence_cap)) {
        return setting(pt, "cap_floor", text(settings.cap_floor)) + "must be finite, within [0, confidence_cap]";
    }
    if (!duplication && settings.cap_floor != 0.0f) {
        return unread(pt, "cap_floor", text(settings.cap_floor), readers);
    }
    if (duplication && !(std::isfinite(settings.cap_exponent) && settings.cap_exponent > 0.0f &&
                         settings.cap_exponent <= 1.0f)) {
        return setting(pt, "cap_exponent", text(settings.cap_exponent)) + "must be finite, within (0, 1]";
    }
    if (!duplication && settings.cap_exponent != 0.0f) {
        return unread(pt, "cap_exponent", text(settings.cap_exponent), readers);
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> invalid(const Schedule& schedule) {
    const std::vector<PassKind>& passes = schedule.passes;
    if (passes.empty()) {
        return "the frame graph has no passes";
    }
    for (const auto broken :
         {accumulation_broken, presentation_broken, radiance_broken, restir_pt_broken, denoise_broken}) {
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

    const bool denoising = std::ranges::find(passes, PassKind::denoise) != passes.end();
    if (denoising && !schedule.denoise) {
        return "the frame graph has a denoise pass and no [denoise] settings";
    }
    if (!denoising && schedule.denoise) {
        return "the frame graph has [denoise] settings and no denoise pass";
    }
    if (schedule.denoise && schedule.denoise->weights.empty()) {
        return "the denoiser's weights name no directory";
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
        if (std::optional<std::string> reason = invalid(*schedule.restir_pt, passes)) {
            return reason;
        }
    }

    // A denoised graph's light is traced checkerboarded: a sigma the traced
    // lattice pairs at (schedule.h).
    const float least_sigma = contracts::min_pairing_sigma * contracts::checkerboard_lattice_step;
    if (denoising && std::ranges::find(passes, PassKind::restir_pt_spatial) != passes.end() &&
        schedule.restir_pt->sigma < least_sigma) {
        return setting(pt, "sigma", text(schedule.restir_pt->sigma)) + "must be at least " + text(least_sigma) +
               " pixels in a denoised graph: its light is traced checkerboarded, and the traced field's lattice "
               "pairs at sigma over its step, " +
               text(contracts::checkerboard_lattice_step) + " pixels, at least " + text(contracts::min_pairing_sigma);
    }
    return std::nullopt;
}

std::optional<std::string> invalid(const passes::RestirPt& settings, std::span<const PassKind> passes) {
    const auto has = [&](PassKind kind) { return std::ranges::find(passes, kind) != passes.end(); };
    const bool temporal = has(PassKind::restir_pt_temporal);
    if (std::optional<std::string> reason = pt_paths_broken(settings)) {
        return reason;
    }
    if (std::optional<std::string> reason = pt_spatial_broken(settings, has(PassKind::restir_pt_spatial))) {
        return reason;
    }
    if (std::optional<std::string> reason = pt_temporal_broken(settings, temporal)) {
        return reason;
    }
    return pt_duplication_broken(settings, temporal, has(PassKind::restir_pt_duplication));
}

std::optional<std::string> invalid(const passes::ToneMap& settings) {
    if (!std::isfinite(settings.exposure) || std::abs(settings.exposure) > passes::max_exposure) {
        const std::string limit = text(passes::max_exposure);
        return "the tone map's exposure, " + text(settings.exposure) + " stops, must be finite, within [-" + limit +
               ", " + limit + "]";
    }
    if (!std::isfinite(settings.bloom) || settings.bloom < 0.0f || settings.bloom >= 1.0f) {
        return "the tone map's bloom, " + text(settings.bloom) + ", must be within [0, 1)";
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

bool presents(const Schedule& schedule) {
    return std::ranges::any_of(schedule.passes, [](PassKind kind) { return writes_target(kind); });
}

bool writes_features(const Schedule& schedule) {
    return std::ranges::any_of(schedule.passes, [](PassKind kind) { return writes_features(kind); });
}

bool reuses_history(const Schedule& schedule) {
    return std::ranges::any_of(schedule.passes, [](PassKind kind) {
        return kind == PassKind::restir_pt_temporal || kind == PassKind::denoise;
    });
}

bool denoises(const Schedule& schedule) {
    return std::ranges::find(schedule.passes, PassKind::denoise) != schedule.passes.end();
}

Extent light_size(const Schedule& schedule, Extent output) {
    if (!denoises(schedule)) {
        return output;
    }
    const auto halved = [](std::uint32_t side) {
        return side / contracts::denoise_scale + (side % contracts::denoise_scale != 0 ? 1u : 0u);
    };
    return {halved(output.width), halved(output.height)};
}

}  // namespace serenity::frame
