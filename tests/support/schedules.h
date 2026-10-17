#pragma once

// What the tests that build frame graphs share (ES.3, F.10): a tone map's
// look, why the core refuses a schedule, and ReSTIR PT's settings at the
// values graphs/ are written with (core/passes/restir_pt.h, written_*), for
// the passes that read them and zero for the
// rest, as the core requires (core/frame/schedule.h). Built from the
// constants, not written out again, so a test that says it runs at the
// written defaults does, whatever they become.

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"
#include "core/passes/restir_pt.h"
#include "core/passes/tone_map.h"
#include "support/text.h"

namespace serenity::tests {

// graphs/path.toml's tone map: no exposure, a little bloom.
inline constexpr passes::ToneMap look{0.0f, 0.04f, {0.0f, 0.0f}};

// Why the core refuses `schedule`, or "" if it accepts it.
inline std::string refusal(const frame::Schedule& schedule) {
    return frame::invalid(schedule).value_or("");
}

// Whether the core refuses `schedule` for a reason that says `part`.
inline bool refused_for(const frame::Schedule& schedule, std::string_view part) {
    const std::string reason = refusal(schedule);
    INFO(reason);
    return contains(reason, part);
}

// Whether `kinds` holds `kind` (std::ranges::contains is C++23's).
inline bool has(const std::vector<frame::PassKind>& kinds, frame::PassKind kind) {
    return std::ranges::find(kinds, kind) != kinds.end();
}

// ReSTIR PT's settings for a graph of passes `kinds`: the written value of each
// setting they read, 0 for the rest.
inline passes::RestirPt written_restir_pt(const std::vector<frame::PassKind>& kinds) {
    passes::RestirPt s{};
    s.nee_candidates = passes::written_nee_candidates;
    s.max_surfaces = passes::written_max_surfaces;
    s.footprint = passes::written_footprint;
    s.roughness = passes::written_roughness;
    if (has(kinds, frame::PassKind::restir_pt_spatial)) {
        s.neighbors = passes::written_path_neighbors;
        s.sigma = passes::written_sigma;
    }
    if (has(kinds, frame::PassKind::restir_pt_temporal)) {
        s.confidence_cap = passes::written_confidence_cap;
    }
    if (has(kinds, frame::PassKind::restir_pt_duplication)) {
        s.cap_floor = passes::written_cap_floor;
        s.cap_exponent = passes::written_cap_exponent;
    }
    return s;
}

// ReSTIR PT's passes as graphs/restir_pt.toml runs them.
inline const std::vector<frame::PassKind> restir_pt_passes = {
    frame::PassKind::restir_pt_initial, frame::PassKind::restir_pt_temporal, frame::PassKind::restir_pt_spatial,
    frame::PassKind::restir_pt_shade, frame::PassKind::restir_pt_duplication};

// A ReSTIR PT graph of `kinds`, its light shown as it is (display), its
// settings the written ones.
inline frame::Schedule restir_pt_graph(const std::vector<frame::PassKind>& kinds) {
    frame::Schedule s{kinds, std::nullopt};
    s.passes.push_back(frame::PassKind::display);
    s.restir_pt = written_restir_pt(kinds);
    return s;
}

}  // namespace serenity::tests
