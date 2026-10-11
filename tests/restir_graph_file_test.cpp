// [restir_di] in frame graph files (core/frame/graph_file.h): every key a
// graph's passes read is required, every key it does not read refused, and
// every value checked at its own line; graphs/restir_ris.toml reads as the
// RIS graph at RTXDI's defaults.

#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/passes/restir_di.h"
#include "support/text.h"
#include "test_paths.h"

using serenity::frame::GraphFileError;
using serenity::frame::load_schedule;
using serenity::frame::parse_schedule;
using serenity::frame::PassKind;
using serenity::passes::BiasCorrection;
using serenity::passes::Permutation;
using serenity::passes::VisibilityReuse;
using serenity::tests::contains;
using serenity::tests::graphs_dir;

namespace {

std::string error_for(const std::string& text) {
    return serenity::tests::error_of<GraphFileError>([&] { return parse_schedule(text, "graph.toml"); });
}

// Line 1 the passes, line 2 the table's header, the keys from line 3.
constexpr std::string_view ris = "passes = [\"restir_initial\", \"restir_shade\", \"display\"]\n[restir_di]\n";
constexpr std::string_view both =
    "passes = [\"restir_initial\", \"restir_temporal\", \"restir_spatial\", \"restir_shade\", \"display\"]\n"
    "[restir_di]\n";

// Every key both merges read, at the values the graphs are written with,
// one a line from line 3: candidates 3, history 4, permutation 5, neighbors
// 6, radius 7, depth_threshold 8, normal_threshold 9, bias_correction 10,
// visibility_reuse 11, visibility_age 12, visibility_distance 13.
constexpr std::string_view every_key =
    "candidates = 8\nhistory = 20\npermutation = true\nneighbors = 1\nradius = 32.0\ndepth_threshold = 0.1\n"
    "normal_threshold = 0.5\nbias_correction = \"basic\"\nvisibility_reuse = true\nvisibility_age = 4\n"
    "visibility_distance = 16.0\n";

std::string both_with(std::string_view from, std::string_view to) {
    return std::string(both) + serenity::tests::replaced(std::string(every_key), from, to);
}

}  // namespace

TEST_CASE("the RIS graph in graphs/ reads, at RTXDI's candidates") {
    const auto schedule = load_schedule(graphs_dir / "restir_ris.toml");
    REQUIRE(schedule.passes.size() == 3);
    CHECK(schedule.passes[0] == PassKind::restir_initial);
    CHECK(schedule.passes[1] == PassKind::restir_shade);
    CHECK(schedule.passes[2] == PassKind::tone_map);
    REQUIRE(schedule.restir_di.has_value());
    CHECK(schedule.restir_di->candidates == serenity::passes::written_candidates);
    CHECK(schedule.restir_di->history == 0u);
    CHECK(schedule.restir_di->bias_correction == BiasCorrection::basic);
    CHECK(schedule.restir_di->visibility_reuse == VisibilityReuse::off);
    REQUIRE(schedule.tone_map.has_value());
}

TEST_CASE("every key of a graph with both merges is read, as written") {
    const auto schedule = parse_schedule(std::string(both) + std::string(every_key), "graph.toml");
    REQUIRE(schedule.restir_di.has_value());
    const auto& s = *schedule.restir_di;
    CHECK(s.candidates == 8u);
    CHECK(s.history == 20u);
    CHECK(s.permutation == Permutation::on);
    CHECK(s.neighbors == 1u);
    CHECK(s.radius == 32.0f);
    CHECK(s.depth_threshold == 0.1f);
    CHECK(s.normal_threshold == 0.5f);
    CHECK(s.bias_correction == BiasCorrection::basic);
    CHECK(s.visibility_reuse == VisibilityReuse::on);
    CHECK(s.visibility_age == 4u);
    CHECK(s.visibility_distance == 16.0f);

    // Raytraced: the final-visibility keys are not read, and so not there.
    const auto raytraced = parse_schedule(
        std::string(both) + "candidates = 8\nhistory = 20\npermutation = false\nneighbors = 1\nradius = 32.0\n"
                            "depth_threshold = 0.1\nnormal_threshold = 0.5\nbias_correction = \"raytraced\"\n",
        "graph.toml");
    CHECK(raytraced.restir_di->bias_correction == BiasCorrection::raytraced);
    CHECK(raytraced.restir_di->permutation == Permutation::off);
    // Basic with reuse off: age and distance not read.
    const auto unreused = parse_schedule(
        std::string(both) + "candidates = 8\nhistory = 20\npermutation = true\nneighbors = 1\nradius = 32.0\n"
                            "depth_threshold = 0.1\nnormal_threshold = 0.5\nbias_correction = \"basic\"\n"
                            "visibility_reuse = false\n",
        "graph.toml");
    CHECK(unreused.restir_di->visibility_reuse == VisibilityReuse::off);
    CHECK(unreused.restir_di->visibility_age == 0u);
}

TEST_CASE("a [restir_di] key the graph's passes do not read is an error at its line") {
    CHECK(contains(error_for(std::string(ris) + "candidates = 8\nhistory = 20\n"),
                   "graph.toml:4: 'history' is not read by this frame graph's passes"));
    CHECK(contains(error_for(std::string(ris) + "candidates = 8\nbias_correction = \"basic\"\n"),
                   "graph.toml:4: 'bias_correction' is not read"));
    CHECK(contains(error_for(std::string(ris) + "candidates = 8\nglare = 1\n"),
                   "graph.toml:4: unknown key 'glare' in [restir_di]"));
    // Unknown in a graph that reads every key.
    CHECK(contains(error_for(both_with("visibility_distance = 16.0\n", "visibility_distance = 16.0\nglare = 1\n")),
                   "graph.toml:14: unknown key 'glare' in [restir_di]"));
    // Raytraced mode reads no final-visibility key; the first in the file
    // is named, though toml++ keeps the keys sorted.
    CHECK(contains(error_for(both_with("\"basic\"", "\"raytraced\"")), "graph.toml:11: 'visibility_reuse' is not read"));
    // Reuse off reads no age.
    CHECK(contains(error_for(both_with("visibility_reuse = true", "visibility_reuse = false")),
                   "graph.toml:12: 'visibility_age' is not read"));
}

TEST_CASE("a [restir_di] key the graph's passes read is required") {
    CHECK(contains(error_for(std::string(ris)), "graph.toml:2: [restir_di] has no 'candidates'"));
    CHECK(contains(error_for(both_with("history = 20\n", "")), "[restir_di] has no 'history'"));
    CHECK(contains(error_for(both_with("bias_correction = \"basic\"\n", "")), "[restir_di] has no 'bias_correction'"));
    CHECK(contains(error_for(both_with("visibility_distance = 16.0\n", "")),
                   "[restir_di] has no 'visibility_distance'"));
}

TEST_CASE("every [restir_di] value of the wrong type or out of range is an error at its line") {
    CHECK(contains(error_for(both_with("candidates = 8", "candidates = 8.0")), "graph.toml:3: 'candidates' must be an integer"));
    CHECK(contains(error_for(both_with("candidates = 8", "candidates = 0")),
                   "graph.toml:3: ReSTIR DI's candidates, 0, must be an integer in [1, 32]"));
    CHECK(contains(error_for(both_with("candidates = 8", "candidates = -1")), "graph.toml:3: 'candidates', -1, is no count"));
    CHECK(contains(error_for(both_with("candidates = 8", "candidates = 5000000000")),
                   "graph.toml:3: 'candidates', 5000000000, is no count"));
    CHECK(contains(error_for(both_with("history = 20", "history = 65")), "graph.toml:4: ReSTIR DI's history, 65"));
    CHECK(contains(error_for(both_with("permutation = true", "permutation = 1")),
                   "graph.toml:5: 'permutation' must be true or false"));
    CHECK(contains(error_for(both_with("neighbors = 1", "neighbors = 33")), "graph.toml:6: ReSTIR DI's neighbors, 33"));
    CHECK(contains(error_for(both_with("radius = 32.0", "radius = 0.5")), "graph.toml:7: ReSTIR DI's radius, 0.5"));
    CHECK(contains(error_for(both_with("radius = 32.0", "radius = \"far\"")), "graph.toml:7: 'radius' must be a number"));
    CHECK(contains(error_for(both_with("radius = 32.0", "radius = inf")), "graph.toml:7: 'radius' must be a finite"));
    CHECK(contains(error_for(both_with("depth_threshold = 0.1", "depth_threshold = 0.0")),
                   "graph.toml:8: ReSTIR DI's depth_threshold, 0"));
    CHECK(contains(error_for(both_with("normal_threshold = 0.5", "normal_threshold = -2.0")),
                   "graph.toml:9: ReSTIR DI's normal_threshold, -2"));
    CHECK(contains(error_for(both_with("\"basic\"", "\"unbiased\"")),
                   "graph.toml:10: 'bias_correction' must be \"basic\" or \"raytraced\""));
    CHECK(contains(error_for(both_with("visibility_reuse = true", "visibility_reuse = \"yes\"")),
                   "graph.toml:11: 'visibility_reuse' must be true or false"));
    CHECK(contains(error_for(both_with("visibility_age = 4", "visibility_age = 256")),
                   "graph.toml:12: ReSTIR DI's visibility_age, 256"));
    CHECK(contains(error_for(both_with("visibility_distance = 16.0", "visibility_distance = 200.0")),
                   "graph.toml:13: ReSTIR DI's visibility_distance, 200"));
}

TEST_CASE("[restir_di] comes exactly with ReSTIR DI's passes, and they in their order") {
    CHECK(contains(error_for("passes = [\"restir_initial\", \"restir_shade\", \"display\"]\n"),
                   "graph.toml:1: the frame graph has ReSTIR DI's passes and no [restir_di] settings"));
    CHECK(contains(error_for("passes = [\"path\", \"display\"]\n[restir_di]\ncandidates = 8\n"),
                   "graph.toml:2: the frame graph has [restir_di] settings and no ReSTIR DI pass"));
    CHECK(contains(error_for("passes = [\"restir_initial\", \"restir_shade\", \"display\"]\nrestir_di = 1\n"),
                   "graph.toml:2: 'restir_di' must be a table"));
    CHECK(contains(error_for("passes = [\"restir_initial\", \"restir_spatial\", \"restir_temporal\", \"restir_shade\", "
                             "\"display\"]\n[restir_di]\n" +
                             std::string(every_key)),
                   "graph.toml:1: ReSTIR DI's passes run in the order"));
}
