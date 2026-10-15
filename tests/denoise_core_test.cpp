// The denoiser in the core: its pass's rules and two sizes (core/frame/
// schedule.h), its graph settings (core/frame/graph_file.h), the JSON its
// export's layout is read with (core/denoise/json.h), and the network read
// from an export (core/denoise/network.h), on the test fixture's network
// (tests/data/denoise/network, training/denoise_check.py) and on copies
// of it broken one way at a time.

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/denoise.h"
#include "core/denoise/json.h"
#include "core/denoise/network.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "support/text.h"
#include "test_paths.h"

using namespace serenity;
using frame::PassKind;
using frame::Schedule;

namespace {

const std::vector<PassKind> denoised = {PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::features,
                                        PassKind::denoise, PassKind::tone_map};

Schedule schedule_of(std::vector<PassKind> passes, bool settings = true) {
    Schedule s{std::move(passes), passes::ToneMap{0.0f, 0.04f, {}}};
    s.restir_pt = passes::RestirPt{};
    s.restir_pt->nee_candidates = 32;
    s.restir_pt->max_surfaces = 9;
    s.restir_pt->footprint = 0.02f;
    s.restir_pt->roughness = 0.2f;
    if (settings) {
        s.denoise = passes::Denoise{"network"};
    }
    return s;
}

std::string refusal(const Schedule& s) {
    return frame::invalid(s).value_or("");
}

std::string file_text(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<std::byte> file_bytes(const std::filesystem::path& path) {
    const std::string text = file_text(path);
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

const std::filesystem::path fixture_network = tests::data_dir / "denoise" / "network";

// The fixture network's layout with `from` replaced by `to` once, parsed
// with its weights; the NetworkError's message, or "" if it reads.
std::string network_refusal(const std::string& from, const std::string& to,
                            std::optional<std::size_t> weights_size = std::nullopt) {
    std::string layout = file_text(fixture_network / "layout.json");
    const std::size_t at = layout.find(from);
    REQUIRE(at != std::string::npos);
    layout.replace(at, from.size(), to);
    std::vector<std::byte> weights = file_bytes(fixture_network / "weights.bin");
    if (weights_size) {
        weights.resize(*weights_size);
    }
    return tests::error_of<denoise::NetworkError>(
        [&] { return denoise::parse_network(layout, weights, "fixture"); });
}

}  // namespace

TEST_CASE("the denoise pass reads the light and features and writes the radiance its presenting pass shows") {
    CHECK(frame::reads_radiance(PassKind::denoise));
    CHECK_FALSE(frame::writes_radiance(PassKind::denoise));
    CHECK_FALSE(frame::writes_target(PassKind::denoise));
    CHECK_FALSE(frame::accumulates(PassKind::denoise));
    CHECK(frame::pass_kind("denoise") == PassKind::denoise);
    const Schedule s = schedule_of(denoised);
    CHECK(refusal(s).empty());
    CHECK(frame::denoises(s));
    CHECK(frame::reuses_history(s));
}

TEST_CASE("a denoised graph's light is half its output each way, rounded up; any other graph's is its output") {
    const Schedule s = schedule_of(denoised);
    CHECK(frame::light_size(s, {3456, 2234}) == frame::Extent{1728, 1117});
    CHECK(frame::light_size(s, {1920, 1080}) == frame::Extent{960, 540});
    CHECK(frame::light_size(s, {1921, 1}) == frame::Extent{961, 1});
    const Schedule plain{{PassKind::path, PassKind::display}, std::nullopt};
    CHECK(frame::light_size(plain, {1921, 1}) == frame::Extent{1921, 1});
    CHECK_FALSE(frame::reuses_history(plain));
}

TEST_CASE("the denoise pass's rules") {
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::denoise,
                                               PassKind::tone_map})),
                          "no features pass comes before it"));
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade,
                                               PassKind::features, PassKind::denoise, PassKind::denoise,
                                               PassKind::tone_map})),
                          "denoise more than once"));
    Schedule preview{{PassKind::preview, PassKind::features, PassKind::denoise, PassKind::display}, std::nullopt};
    preview.denoise = passes::Denoise{"network"};
    CHECK(tests::contains(refusal(preview), "no light pass that accumulates"));
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_temporal,
                                               PassKind::restir_pt_shade, PassKind::features, PassKind::denoise,
                                               PassKind::restir_pt_duplication, PassKind::tone_map})),
                          "directly before"));
    CHECK(tests::contains(refusal(schedule_of(denoised, false)), "no [denoise] settings"));
    Schedule stray = schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::tone_map});
    CHECK(tests::contains(refusal(stray), "[denoise] settings and no denoise pass"));
}

TEST_CASE("the denoised graph in graphs/ reads, its weights relative to its own directory") {
    const Schedule s = frame::load_schedule(tests::graphs_dir / "restir_pt_denoised.toml");
    CHECK(s.passes.back() == PassKind::tone_map);
    CHECK(s.passes[s.passes.size() - 2] == PassKind::denoise);
    REQUIRE(s.denoise.has_value());
    CHECK(s.denoise->weights == tests::graphs_dir / "../../serenity-data/denoiser-v1-model/wide");
}

TEST_CASE("every mistake in [denoise] is an error") {
    const std::string passes =
        "passes = [\"path\", \"features\", \"denoise\", \"tone_map\"]\n[tone_map]\nexposure = 0.0\nbloom = 0.04\n";
    const auto error = [&](const std::string& rest) {
        return tests::error_of<frame::GraphFileError>([&] { return frame::parse_schedule(passes + rest, "g.toml"); });
    };
    CHECK(error("[denoise]\nweights = \"a\"\n").empty());
    CHECK(tests::contains(error(""), "no [denoise] settings"));
    CHECK(tests::contains(error("[denoise]\n"), "weights"));
    CHECK(tests::contains(error("[denoise]\nweights = 3\n"), "'weights' must name"));
    CHECK(tests::contains(error("[denoise]\nweights = \"\"\n"), "'weights' must name"));
    CHECK(tests::contains(error("[denoise]\nweights = \"a\"\nscale = 2\n"), "scale"));
    CHECK(tests::contains(tests::error_of<frame::GraphFileError>([] {
                              return frame::parse_schedule(
                                  "passes = [\"path\", \"display\"]\n[denoise]\nweights = \"a\"\n", "g.toml");
                          }),
                          "no denoise pass"));
}

TEST_CASE("JSON is read whole and strictly") {
    const denoise::Json doc = denoise::parse_json(R"( {"a": [1, -2.5e1, true, null, "x\n\u0041"], "b": {}} )");
    const denoise::JsonArray& a = doc.at("a").array("a");
    REQUIRE(a.size() == 5);
    CHECK(a[0].number("") == 1.0);
    CHECK(a[1].number("") == -25.0);
    CHECK(a[4].string("") == "x\nA");
    for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "[1] 2", "\"\\u00e9\"", "tru"}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(denoise::parse_json(bad), denoise::JsonError);
    }
    CHECK_THROWS_AS(doc.at("c"), denoise::JsonError);
    CHECK_THROWS_AS(doc.at("a").number("a"), denoise::JsonError);
}

TEST_CASE("the fixture's exported network reads, its widths and tensors as the structure needs") {
    const denoise::Network n = denoise::load_network(fixture_network);
    CHECK(n.widths.enc == std::array<std::uint32_t, 4>{8, 16, 24, 32});
    CHECK(n.widths.bottleneck == 40);
    CHECK(n.widths.dec == std::array<std::uint32_t, 4>{32, 24, 16, 8});
    CHECK(n.enc[0].inputs == contracts::network_inputs);
    CHECK(n.enc[0].weight.size() == std::size_t{8} * 28 * 9);
    CHECK(n.dec[0].inputs == 40 + 32);  // the bottleneck's, then the skip's
    CHECK(n.head.outputs == contracts::head_outputs);
    CHECK(n.head.weight.size() == std::size_t{124} * 8);
}

TEST_CASE("a network export that is not what the renderer runs is refused, naming what") {
    CHECK(network_refusal("\"format\": \"serenity-denoiser-weights/1\"", "\"format\": \"other\"").find("format") !=
          std::string::npos);
    CHECK(tests::contains(network_refusal("\"albedo_floor\": 0.02", "\"albedo_floor\": 0.03"), "albedo_floor"));
    CHECK(tests::contains(network_refusal("\"masked_tap_score\": -10000.0", "\"masked_tap_score\": -1000.0"),
                          "masked_tap_score"));
    CHECK(tests::contains(network_refusal("\"bottleneck\": 40", "\"bottleneck\": 41"), "multiple of 8"));
    CHECK(tests::contains(network_refusal("\"unet.head.bias\"", "\"unet.head.extra\""), "no tensor unet.head.bias"));
    CHECK(tests::contains(network_refusal("{", "{", 1000), "inside weights.bin"));
    CHECK(tests::contains(network_refusal("{", "{ ,"), "layout.json"));
    CHECK(network_refusal("{", "{").empty());
}
