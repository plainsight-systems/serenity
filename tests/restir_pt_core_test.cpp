// ReSTIR PT's core: its graph and settings (core/frame/graph_file.h,
// schedule.h, core/passes/restir_pt.h), and the reuse textures of its
// paired spatial reuse (core/sampling/pairing.h): each an involution, its
// offsets' standard deviation sigma, the same for the same seed.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "core/passes/restir_pt.h"
#include "core/sampling/pairing.h"
#include "support/text.h"
#include "test_paths.h"

using serenity::frame::GraphFileError;
using serenity::frame::load_schedule;
using serenity::frame::parse_schedule;
using serenity::frame::PassKind;
using serenity::tests::contains;
using serenity::tests::graphs_dir;

namespace {

std::string error_for(const std::string& text) {
    return serenity::tests::error_of<GraphFileError>([&] { return parse_schedule(text, "graph.toml"); });
}

constexpr std::string_view initial_keys = "nee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\n";

}  // namespace

TEST_CASE("graphs/restir_pt.toml reads as ReSTIR PT Enhanced at its defaults, with MNEE") {
    const auto schedule = load_schedule(graphs_dir / "restir_pt.toml");
    REQUIRE(schedule.passes.size() == 6);
    CHECK(schedule.passes[0] == PassKind::restir_pt_initial);
    CHECK(schedule.passes[1] == PassKind::restir_pt_temporal);
    CHECK(schedule.passes[2] == PassKind::restir_pt_spatial);
    CHECK(schedule.passes[3] == PassKind::restir_pt_shade);
    CHECK(schedule.passes[4] == PassKind::restir_pt_duplication);
    CHECK(schedule.passes[5] == PassKind::tone_map);
    CHECK(serenity::frame::reuses_history(schedule));
    REQUIRE(schedule.restir_pt.has_value());
    CHECK(schedule.restir_pt->nee_candidates == serenity::passes::written_nee_candidates);
    CHECK(schedule.restir_pt->max_surfaces == serenity::passes::written_max_surfaces);
    CHECK(schedule.restir_pt->footprint == serenity::passes::written_footprint);
    CHECK(schedule.restir_pt->roughness == serenity::passes::written_roughness);
    CHECK(schedule.restir_pt->neighbors == serenity::passes::written_path_neighbors);
    CHECK(schedule.restir_pt->sigma == serenity::passes::written_sigma);
    CHECK(schedule.restir_pt->confidence_cap == serenity::passes::written_confidence_cap);
    CHECK(schedule.restir_pt->cap_floor == serenity::passes::written_cap_floor);
    CHECK(schedule.restir_pt->cap_exponent == serenity::passes::written_cap_exponent);
    CHECK(schedule.mnee);
    CHECK_FALSE(schedule.restir_di.has_value());
}

TEST_CASE("ReSTIR PT's temporal reuse and duplication map: their order, their settings, and the map only with "
          "temporal reuse") {
    constexpr std::string_view temporal_keys = "confidence_cap = 20.0\n";
    constexpr std::string_view duplication_keys = "cap_floor = 1.0\ncap_exponent = 0.1\n";
    // Temporal reuse alone reads; it reuses history; the duplication map
    // after the shade with it reads.
    const auto alone = parse_schedule("passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_shade\", "
                                      "\"display\"]\n[restir_pt]\n" +
                                          std::string(initial_keys) + std::string(temporal_keys),
                                      "graph.toml");
    CHECK(serenity::frame::reuses_history(alone));
    REQUIRE(alone.restir_pt.has_value());
    CHECK(alone.restir_pt->confidence_cap == 20.0f);
    CHECK(alone.restir_pt->cap_exponent == 0.0f);
    // The map without temporal reuse, whose cap is all it serves.
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"restir_pt_duplication\", "
                             "\"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + std::string(duplication_keys)),
                   "without restir_pt_temporal"));
    // Temporal reuse after the spatial: out of order.
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_spatial\", \"restir_pt_temporal\", "
                             "\"restir_pt_shade\", \"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + "neighbors = 3\nsigma = 16.0\n" + std::string(temporal_keys)),
                   "in the order"));
    // A cap without temporal reuse, and a floor without the map: keys nobody reads.
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + std::string(temporal_keys)),
                   "read only with restir_pt_temporal"));
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_shade\", "
                             "\"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + std::string(temporal_keys) + "cap_floor = 1.0\n"),
                   "read only with restir_pt_duplication"));
    // Out of range: a cap past 60, a floor above the cap, an exponent of 0.
    const std::string with_map = "passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_shade\", "
                                 "\"restir_pt_duplication\", \"display\"]\n[restir_pt]\n" +
                                 std::string(initial_keys);
    CHECK(contains(error_for(with_map + "confidence_cap = 61.0\n" + std::string(duplication_keys)), "confidence_cap"));
    CHECK(contains(error_for(with_map + "confidence_cap = 10.0\ncap_floor = 11.0\ncap_exponent = 0.1\n"),
                   "cap_floor"));
    CHECK(contains(error_for(with_map + std::string(temporal_keys) + "cap_floor = 1.0\ncap_exponent = 0.0\n"),
                   "cap_exponent"));
    // A floor up to a cap the graph raised is taken.
    const auto raised = parse_schedule(with_map + "confidence_cap = 40.0\ncap_floor = 30.0\ncap_exponent = 0.5\n",
                                       "graph.toml");
    REQUIRE(raised.restir_pt.has_value());
    CHECK(raised.restir_pt->cap_floor == 30.0f);
}

TEST_CASE("a graph runs one ReSTIR: ReSTIR PT's passes are refused beside ReSTIR DI's") {
    const std::string error =
        error_for("passes = [\"restir_initial\", \"restir_pt_initial\", \"restir_pt_shade\", "
                  "\"display\"]\n[restir_di]\ncandidates = 8\nbsdf_candidates = 1\n[restir_pt]\n" +
                  std::string(initial_keys));
    INFO(error);
    CHECK(contains(error, "one family"));
}

TEST_CASE("ReSTIR PT's passes keep their order, and its settings come with them, in range, at their own line") {
    CHECK(contains(error_for("passes = [\"restir_pt_shade\", \"restir_pt_initial\", \"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys)),
                   "in the order"));
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"display\"]\n"),
                   "no [restir_pt] settings"));
    // Line 3 is nee_candidates.
    const std::string many = error_for("passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"display\"]\n"
                                       "[restir_pt]\nnee_candidates = 33\nmax_surfaces = 9\nfootprint = 0.02\n"
                                       "roughness = 0.2\n");
    INFO(many);
    CHECK(contains(many, "graph.toml:3:"));
    CHECK(contains(many, "nee_candidates"));
    // Neighbors without the spatial pass: a key nobody reads.
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + "neighbors = 3\n"),
                   "read only with restir_pt_spatial"));
    CHECK(contains(error_for("passes = [\"restir_pt_initial\", \"restir_pt_spatial\", \"restir_pt_shade\", "
                             "\"display\"]\n[restir_pt]\n" +
                             std::string(initial_keys) + "neighbors = 3\nsigma = 0.5\n"),
                   "sigma"));
}

TEST_CASE("a path reservoir's flags read back as packed") {
    using namespace serenity::contracts;
    const PathShape shape{7u, 10u, 4u, PathEnd::struck, lobe_reflection | lobe_glossy, lobe_reflection | lobe_diffuse,
                          5u};
    const std::uint32_t flags = pack_path_flags(shape);
    CHECK(path_count(flags) == 7u);
    CHECK(path_length(flags) == 10u);
    CHECK(path_rc(flags) == 4u);
    CHECK(path_end(flags) == PathEnd::struck);
    CHECK(path_lobe_before(flags) == (lobe_reflection | lobe_glossy));
    CHECK(path_lobe_at(flags) == (lobe_reflection | lobe_diffuse));
    CHECK(path_face(flags) == 5u);
    CHECK((flags >> 31u) == 0u);
}

TEST_CASE("eq. 3's shuffle count") {
    CHECK(serenity::sampling::shuffle_count(16.0) == 128u);
    // The least sigma, as the graph holds it: one shuffle (the paper's footnote).
    CHECK(serenity::sampling::shuffle_count(static_cast<double>(serenity::contracts::min_pairing_sigma)) == 1u);
    CHECK_THROWS_AS(serenity::sampling::shuffle_count(0.5), std::invalid_argument);
}

TEST_CASE("a reuse texture pairs every texel with one other, both ways, at a standard deviation of sigma") {
    using serenity::contracts::pairing_widths;
    for (const std::uint32_t width : pairing_widths) {
        const std::vector<std::uint16_t> texture = serenity::sampling::reuse_texture(width, 16.0, 7u);
        REQUIRE(texture.size() == std::size_t{width} * width);
        const int w = static_cast<int>(width);
        double squares = 0.0;
        std::size_t self = 0;
        for (int y = 0; y < w; ++y) {
            for (int x = 0; x < w; ++x) {
                const auto d = serenity::sampling::unpack_offset(texture[static_cast<std::size_t>(y * w + x)]);
                self += (d.x == 0 && d.y == 0) ? 1u : 0u;
                // The partner, wrapped, reads the negation: an involution.
                const int px = ((x + d.x) % w + w) % w;
                const int py = ((y + d.y) % w + w) % w;
                const auto back = serenity::sampling::unpack_offset(texture[static_cast<std::size_t>(py * w + px)]);
                REQUIRE(back.x == -d.x);
                REQUIRE(back.y == -d.y);
                squares += static_cast<double>(d.x * d.x + d.y * d.y);
            }
        }
        CHECK(self == 0u);
        // Per axis: the mean square over both axes, halved.
        const double deviation = std::sqrt(squares / (2.0 * static_cast<double>(texture.size())));
        INFO("width " << width << ": deviation " << deviation);
        CHECK(deviation == doctest::Approx(16.0).epsilon(0.1));
    }
}

TEST_CASE("the reuse textures are the same for the same seed, and another for another") {
    const auto a = serenity::sampling::reuse_textures(16.0, 3u);
    CHECK(a == serenity::sampling::reuse_textures(16.0, 3u));
    CHECK(a != serenity::sampling::reuse_textures(16.0, 4u));
    CHECK(a.size() == serenity::contracts::pairing_entries);
}
