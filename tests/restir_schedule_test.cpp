// ReSTIR DI in the core's schedule (core/frame/schedule.h): what its four
// pass kinds read and write, the order a graph keeps them in, and which
// settings each graph's passes read, in what ranges, every other its zero
// (core/contracts/reuse_settings.h).

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"
#include "core/passes/restir_di.h"
#include "support/text.h"

using namespace serenity::frame;
using serenity::contracts::BiasCorrection;
using serenity::contracts::Permutation;
using serenity::contracts::VisibilityReuse;
using serenity::passes::RestirDi;
using serenity::passes::ToneMap;

namespace {

constexpr ToneMap look{0.0f, 0.04f, {0.0f, 0.0f}};

constexpr PassKind initial = PassKind::restir_initial;
constexpr PassKind temporal = PassKind::restir_temporal;
constexpr PassKind spatial = PassKind::restir_spatial;
constexpr PassKind shade = PassKind::restir_shade;

// The settings a graph of `passes` is written with: RTXDI's defaults for
// what its passes read, zeros for the rest (restir_di.h).
RestirDi written_for(const std::vector<PassKind>& passes) {
    const auto has = [&](PassKind kind) {
        for (PassKind p : passes) {
            if (p == kind) {
                return true;
            }
        }
        return false;
    };
    RestirDi s{};
    s.candidates = serenity::passes::written_candidates;
    s.bsdf_candidates = serenity::passes::written_bsdf_candidates;
    if (has(temporal)) {
        s.history = serenity::passes::written_history;
        s.permutation = serenity::passes::written_permutation;
    }
    if (has(spatial)) {
        s.neighbors = serenity::passes::written_neighbors;
        s.radius = serenity::passes::written_radius;
    }
    if (has(temporal) || has(spatial)) {
        s.depth_threshold = serenity::passes::written_depth_threshold;
        s.normal_threshold = serenity::passes::written_normal_threshold;
        s.bias_correction = serenity::passes::written_bias_correction;
        s.visibility_reuse = serenity::passes::written_visibility_reuse;
        s.visibility_age = serenity::passes::written_visibility_age;
        s.visibility_distance = serenity::passes::written_visibility_distance;
    }
    return s;
}

// A ReSTIR graph of `restir` passes, shown through the tone map.
Schedule graph(std::vector<PassKind> restir, std::optional<RestirDi> settings) {
    restir.push_back(PassKind::tone_map);
    return Schedule{restir, look, settings};
}

Schedule written(std::vector<PassKind> restir) {
    const RestirDi settings = written_for(restir);
    return graph(std::move(restir), settings);
}

std::string refusal(const Schedule& schedule) {
    return invalid(schedule).value_or("");
}

bool refused_for(const Schedule& schedule, std::string_view part) {
    const std::string reason = refusal(schedule);
    INFO(reason);
    return serenity::tests::contains(reason, part);
}

// Why the settings rule refuses `settings` for `passes`, or "".
std::string settings_refusal(const RestirDi& settings, const std::vector<PassKind>& passes) {
    return invalid(settings, passes).value_or("");
}

}  // namespace

TEST_CASE("ReSTIR DI's four passes read the scene; the shade alone accumulates and writes radiance") {
    for (const PassKind kind : {initial, temporal, spatial, shade}) {
        CHECK(is_restir(kind));
        CHECK(needs_scene(kind));
        CHECK_FALSE(reads_radiance(kind));
        CHECK_FALSE(writes_target(kind));
        CHECK(pass_kind(name(kind)) == kind);
    }
    CHECK(accumulates(shade));
    CHECK(writes_radiance(shade));
    for (const PassKind kind : {initial, temporal, spatial}) {
        CHECK_FALSE(accumulates(kind));
        CHECK_FALSE(writes_radiance(kind));
    }
    for (const PassKind kind : {PassKind::test_pattern, PassKind::preview, PassKind::path, PassKind::display,
                                PassKind::tone_map}) {
        CHECK_FALSE(is_restir(kind));
    }
    CHECK(name(initial) == "restir_initial");
    CHECK(name(shade) == "restir_shade");
    CHECK_THROWS_AS((void)is_restir(static_cast<PassKind>(9)), std::logic_error);
}

TEST_CASE("ReSTIR DI's graphs, in their order, are valid; only the temporal merge reuses history") {
    for (const std::vector<PassKind>& passes : std::vector<std::vector<PassKind>>{
             {initial, shade}, {initial, temporal, shade}, {initial, spatial, shade}, {initial, temporal, spatial, shade}}) {
        const Schedule schedule = written(passes);
        INFO(refusal(schedule));
        CHECK(refusal(schedule).empty());
        CHECK(accumulates(schedule));
    }
    CHECK_FALSE(reuses_history(written({initial, shade})));
    CHECK_FALSE(reuses_history(written({initial, spatial, shade})));
    CHECK(reuses_history(written({initial, temporal, shade})));
    CHECK_FALSE(reuses_history(Schedule{{PassKind::path, PassKind::tone_map}, look, std::nullopt}));
}

TEST_CASE("ReSTIR DI's passes keep their order, each at most once, the initial pass and the shade both there") {
    CHECK(refused_for(written({initial, spatial, temporal, shade}), "in the order restir_initial, restir_temporal"));
    CHECK(refused_for(written({shade, initial}), "in the order"));
    CHECK(refused_for(written({initial, temporal, temporal, shade}), "restir_temporal more than once"));
    CHECK(refused_for(written({initial, initial, shade}), "restir_initial more than once"));
    // The shade twice is two passes that accumulate, refused by that rule.
    CHECK(refused_for(written({initial, shade, shade}), "passes that accumulate"));
    CHECK(refused_for(Schedule{{initial, temporal, PassKind::path, PassKind::tone_map}, look, written_for({initial, temporal})},
                      "without both restir_initial and restir_shade"));
    // Without the shade nothing writes radiance for the tone map.
    CHECK(refused_for(graph({initial}, written_for({initial})), "no pass before it computes light"));
    CHECK(refused_for(written({temporal, shade}), "without both restir_initial and restir_shade"));
    // Beside another pass that computes light or accumulates.
    CHECK(refused_for(Schedule{{initial, PassKind::path, shade, PassKind::tone_map}, look, written_for({initial})},
                      "passes that accumulate"));
}

TEST_CASE("ReSTIR DI's settings come exactly with its passes") {
    CHECK(refused_for(graph({initial, shade}, std::nullopt), "ReSTIR DI's passes and no [restir_di] settings"));
    CHECK(refused_for(Schedule{{PassKind::path, PassKind::tone_map}, look, written_for({initial})},
                      "[restir_di] settings and no ReSTIR DI pass"));
}

TEST_CASE("the candidates, read by every ReSTIR graph, in [1, 32]") {
    const std::vector<PassKind> passes{initial, shade};
    RestirDi s = written_for(passes);
    for (const std::uint32_t m : {1u, 8u, 32u}) {
        s.candidates = m;
        CHECK(settings_refusal(s, passes).empty());
    }
    for (const std::uint32_t m : {0u, 33u}) {
        s.candidates = m;
        CHECK(serenity::tests::contains(settings_refusal(s, passes), "candidates, " + std::to_string(m) +
                                                                         ", must be an integer in [1, 32]"));
    }
}

TEST_CASE("the BSDF candidates, read by every ReSTIR graph, in [0, 32]") {
    const std::vector<PassKind> passes{initial, shade};
    RestirDi s = written_for(passes);
    for (const std::uint32_t n : {0u, 1u, 32u}) {
        s.bsdf_candidates = n;
        CHECK(settings_refusal(s, passes).empty());
    }
    s.bsdf_candidates = 33u;
    CHECK(serenity::tests::contains(settings_refusal(s, passes), "bsdf_candidates, 33, must be an integer in [0, 32]"));
}

TEST_CASE("each setting the passes do not read must be its zero") {
    const std::vector<PassKind> ris{initial, shade};
    const auto refused = [&](auto change, std::string_view part) {
        RestirDi s = written_for(ris);
        change(s);
        const std::string reason = settings_refusal(s, ris);
        INFO(reason);
        return serenity::tests::contains(reason, part);
    };
    CHECK(refused([](RestirDi& s) { s.history = 20; }, "history, 20, is read only with restir_temporal"));
    CHECK(refused([](RestirDi& s) { s.permutation = Permutation::on; }, "permutation, on, is read only"));
    CHECK(refused([](RestirDi& s) { s.neighbors = 1; }, "neighbors, 1, is read only with restir_spatial"));
    CHECK(refused([](RestirDi& s) { s.radius = 32.0f; }, "radius, 32, is read only with restir_spatial"));
    CHECK(refused([](RestirDi& s) { s.depth_threshold = 0.1f; }, "depth_threshold, 0.1, is read only"));
    CHECK(refused([](RestirDi& s) { s.normal_threshold = 0.5f; }, "normal_threshold, 0.5, is read only"));
    CHECK(refused([](RestirDi& s) { s.bias_correction = BiasCorrection::raytraced; },
                  "bias_correction, raytraced, is read only"));
    CHECK(refused([](RestirDi& s) { s.visibility_reuse = VisibilityReuse::on; }, "visibility_reuse, on, is read only"));
    CHECK(refused([](RestirDi& s) { s.visibility_age = 4; }, "visibility_age, 4, is read only"));
    CHECK(refused([](RestirDi& s) { s.visibility_distance = 16.0f; }, "visibility_distance, 16, is read only"));
}

TEST_CASE("the merges' settings, each in its range") {
    const std::vector<PassKind> both{initial, temporal, spatial, shade};
    const auto reason_for = [&](auto change) {
        RestirDi s = written_for(both);
        change(s);
        return settings_refusal(s, both);
    };
    const auto refused = [&](auto change, std::string_view part) {
        const std::string reason = reason_for(change);
        INFO(reason);
        return serenity::tests::contains(reason, part);
    };
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    CHECK(reason_for([](RestirDi&) {}).empty());
    CHECK(reason_for([](RestirDi& s) { s.history = 64; }).empty());
    CHECK(refused([](RestirDi& s) { s.history = 0; }, "history, 0, must be an integer in [1, 64]"));
    CHECK(refused([](RestirDi& s) { s.history = 65; }, "history, 65, must be an integer in [1, 64]"));
    CHECK(refused([](RestirDi& s) { s.permutation = static_cast<Permutation>(2); }, "permutation, 2, is neither"));
    CHECK(reason_for([](RestirDi& s) { s.neighbors = 32; }).empty());
    CHECK(refused([](RestirDi& s) { s.neighbors = 0; }, "neighbors, 0, must be an integer in [1, 32]"));
    CHECK(refused([](RestirDi& s) { s.neighbors = 33; }, "neighbors, 33"));
    CHECK(reason_for([](RestirDi& s) { s.radius = 1.0f; }).empty());
    CHECK(reason_for([](RestirDi& s) { s.radius = 1024.0f; }).empty());
    CHECK(refused([](RestirDi& s) { s.radius = 0.5f; }, "radius, 0.5, must be finite, within [1, 1024] pixels"));
    CHECK(refused([](RestirDi& s) { s.radius = nan; }, "radius"));
    CHECK(refused([](RestirDi& s) { s.radius = infinity; }, "radius"));
    CHECK(reason_for([](RestirDi& s) { s.depth_threshold = 1.0f; }).empty());
    CHECK(refused([](RestirDi& s) { s.depth_threshold = 0.0f; }, "depth_threshold, 0, must be within (0, 1]"));
    CHECK(refused([](RestirDi& s) { s.depth_threshold = 1.5f; }, "depth_threshold"));
    CHECK(refused([](RestirDi& s) { s.depth_threshold = nan; }, "depth_threshold"));
    CHECK(reason_for([](RestirDi& s) { s.normal_threshold = -1.0f; }).empty());
    CHECK(refused([](RestirDi& s) { s.normal_threshold = 1.01f; }, "normal_threshold, 1.01, must be a cosine"));
    CHECK(refused([](RestirDi& s) { s.normal_threshold = -infinity; }, "normal_threshold"));
    CHECK(refused([](RestirDi& s) { s.bias_correction = static_cast<BiasCorrection>(2); },
                  "bias_correction, 2, is neither basic nor raytraced"));
    CHECK(reason_for([](RestirDi& s) { s.visibility_age = 255; }).empty());
    CHECK(refused([](RestirDi& s) { s.visibility_age = 256; }, "visibility_age, 256, must be an integer in [0, 255]"));
    CHECK(reason_for([](RestirDi& s) { s.visibility_distance = 0.0f; }).empty());
    CHECK(reason_for([](RestirDi& s) { s.visibility_distance = 127.0f; }).empty());
    CHECK(refused([](RestirDi& s) { s.visibility_distance = 128.0f; }, "visibility_distance, 128"));
    CHECK(refused([](RestirDi& s) { s.visibility_distance = -1.0f; }, "visibility_distance"));
    CHECK(refused([](RestirDi& s) { s.visibility_reuse = static_cast<VisibilityReuse>(3); },
                  "visibility_reuse, 3, is neither off nor on"));
}

TEST_CASE("the final-visibility reuse's settings are read only in basic mode, and only with reuse on") {
    const std::vector<PassKind> spatial_graph{initial, spatial, shade};
    RestirDi raytraced = written_for(spatial_graph);
    raytraced.bias_correction = BiasCorrection::raytraced;
    CHECK(serenity::tests::contains(settings_refusal(raytraced, spatial_graph), "visibility_reuse, on, is read only"));
    raytraced.visibility_reuse = VisibilityReuse::off;
    raytraced.visibility_age = 0;
    raytraced.visibility_distance = 0.0f;
    CHECK(settings_refusal(raytraced, spatial_graph).empty());

    RestirDi off = written_for(spatial_graph);
    off.visibility_reuse = VisibilityReuse::off;
    CHECK(serenity::tests::contains(settings_refusal(off, spatial_graph), "visibility_age, 4, is read only"));
    off.visibility_age = 0;
    CHECK(serenity::tests::contains(settings_refusal(off, spatial_graph), "visibility_distance, 16, is read only"));
    off.visibility_distance = 0.0f;
    CHECK(settings_refusal(off, spatial_graph).empty());
}
