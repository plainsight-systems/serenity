// The denoiser in the core: its pass's rules and two sizes (core/frame/
// schedule.h), its graph settings (core/frame/graph_file.h), the JSON its
// export's layout is read with (core/denoise/json.h), and the network read
// from an export (core/denoise/network.h), on the test fixture's network
// (tests/data/denoise/network, training/denoise_check.py) and on copies
// of it broken one way at a time.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/denoise.h"
#include "core/denoise/json.h"
#include "core/denoise/network.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "support/files.h"
#include "support/schedules.h"
#include "support/text.h"
#include "test_paths.h"

using namespace serenity;
using frame::PassKind;
using frame::Schedule;

namespace {

const std::vector<PassKind> denoised = {PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::features,
                                        PassKind::denoise, PassKind::depth_of_field, PassKind::tone_map};

using tests::refusal;

// `kinds`, shown through the tone map, with ReSTIR PT's settings at the
// values graphs/ are written with (support/schedules.h).
Schedule schedule_of(std::vector<PassKind> kinds, bool settings = true) {
    Schedule s{kinds, tests::look};
    s.restir_pt = tests::written_restir_pt(kinds);
    if (settings) {
        s.denoise = passes::Denoise{"network"};
    }
    return s;
}

const std::filesystem::path fixture_network = tests::data_dir / "denoise" / "network";

// The fixture network's layout with `from` replaced by `to` once, parsed
// with its weights; the NetworkError's message, or "" if it reads.
std::string network_refusal(const std::string& from, const std::string& to,
                            std::optional<std::size_t> weights_size = std::nullopt) {
    const std::string layout = tests::replaced(tests::read_bytes(fixture_network / "layout.json"), from, to);
    std::string weights = tests::read_bytes(fixture_network / "weights.bin");
    if (weights_size) {
        weights.resize(*weights_size);
    }
    return tests::error_of<denoise::NetworkError>(
        [&] { return denoise::parse_network(layout, std::as_bytes(std::span(weights)), "fixture"); });
}

}  // namespace

TEST_CASE("the denoise pass reads the light and features and writes the radiance its presenting pass shows") {
    CHECK(frame::reads_radiance(PassKind::denoise));
    CHECK_FALSE(frame::writes_radiance(PassKind::denoise));
    CHECK_FALSE(frame::writes_target(PassKind::denoise));
    CHECK_FALSE(frame::accumulates(PassKind::denoise));
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
    const Schedule plain{{PassKind::path, PassKind::tone_map}, tests::look};
    CHECK(frame::light_size(plain, {1921, 1}) == frame::Extent{1921, 1});
    CHECK_FALSE(frame::reuses_history(plain));
}

TEST_CASE("the denoise pass's rules") {
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::denoise,
                                               PassKind::depth_of_field, PassKind::tone_map})),
                          "no features pass comes before it"));
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade,
                                               PassKind::features, PassKind::denoise, PassKind::denoise,
                                               PassKind::depth_of_field, PassKind::tone_map})),
                          "denoise more than once"));
    // Its light is ReSTIR PT's, traced checkerboarded: the path tracer's is
    // refused.
    Schedule path{{PassKind::path, PassKind::features, PassKind::denoise, PassKind::depth_of_field, PassKind::tone_map},
                  tests::look};
    path.denoise = passes::Denoise{"network"};
    CHECK(tests::contains(refusal(path), "no restir_pt_shade comes before it"));
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_temporal,
                                               PassKind::restir_pt_shade, PassKind::features, PassKind::denoise,
                                               PassKind::restir_pt_duplication, PassKind::depth_of_field,
                                               PassKind::tone_map})),
                          "followed directly by depth_of_field"));
    CHECK(tests::contains(refusal(schedule_of(denoised, false)), "no [denoise] settings"));
    Schedule stray = schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::tone_map});
    CHECK(tests::contains(refusal(stray), "[denoise] settings and no denoise pass"));
}

TEST_CASE("a denoised graph's weights are read relative to its file's directory, an absolute path as it is") {
    const tests::ScratchDirectory scratch("denoise-weights");
    const std::string graph =
        "passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"features\", \"denoise\", \"depth_of_field\", \"tone_map\"]\n"
        "[tone_map]\nexposure = 0.0\nbloom = 0.0\n"
        "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\n";
    std::filesystem::create_directories(scratch / "graphs");
    tests::write_bytes(scratch / "graphs" / "relative.toml", graph + "[denoise]\nweights = \"../model/net\"\n");
    tests::write_bytes(scratch / "graphs" / "absolute.toml", graph + "[denoise]\nweights = \"/model/net\"\n");
    const Schedule relative = frame::load_schedule(scratch / "graphs" / "relative.toml");
    REQUIRE(relative.denoise.has_value());
    CHECK(relative.denoise->weights == scratch / "graphs" / "../model/net");
    const Schedule absolute = frame::load_schedule(scratch / "graphs" / "absolute.toml");
    REQUIRE(absolute.denoise.has_value());
    CHECK(absolute.denoise->weights == "/model/net");
}

TEST_CASE("every mistake in [denoise] is an error") {
    const std::string passes =
        "passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"features\", \"denoise\", \"depth_of_field\", \"tone_map\"]\n"
        "[tone_map]\nexposure = 0.0\nbloom = 0.04\n"
        "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\n";
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
                                  "passes = [\"test_pattern\"]\n[denoise]\nweights = \"a\"\n", "g.toml");
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

TEST_CASE("a JSON object that names a member twice is refused at the second, never read by its first") {
    const auto twice = [] { return denoise::parse_json(R"({"a": 1, "a": 2})"); };
    CHECK(tests::contains(tests::error_of<denoise::JsonError>(twice),
                          "JSON at byte 9: the member name \"a\" a second time in one object"));
    // The same name in two objects is two members.
    CHECK(denoise::parse_json(R"({"a": {"a": 1}})").at("a").at("a").number("") == 1.0);
}

TEST_CASE("the fixture's exported network reads, its widths and tensors as the structure needs") {
    const denoise::Network n = denoise::load_network(fixture_network);
    CHECK(n.widths.enc == std::array<std::uint32_t, 4>{8, 16, 24, 32});
    CHECK(n.widths.bottleneck == 40);
    CHECK(n.widths.dec == std::array<std::uint32_t, 4>{32, 24, 16, 8});
    CHECK(n.enc[0].inputs == contracts::network_inputs);
    CHECK(n.enc[0].weight.size() == std::size_t{n.widths.enc[0]} * contracts::network_inputs * contracts::denoise_taps);
    CHECK(n.dec[0].inputs == 40 + 32);  // the bottleneck's, then the skip's
    CHECK(n.head.outputs == contracts::head_outputs);
    CHECK(n.head.weight.size() == std::size_t{contracts::head_outputs} * n.widths.dec.back());
}

TEST_CASE("a network export that is not what the renderer runs is refused, naming what") {
    CHECK(tests::contains(network_refusal("\"format\": \"serenity-denoiser-weights/1\"", "\"format\": \"other\""),
                          "format"));
    // A network trained before the camera's jitter was an input: 28 inputs,
    // as every export before contract 18's 30 (the demo's
    // denoiser-v2-model among them).
    CHECK(tests::contains(network_refusal("  \"channels\": 30,", "  \"channels\": 28,"),
                          "inputs.channels is not the 30 this renderer runs with"));
    CHECK(tests::contains(network_refusal("\"albedo_floor\": 0.02", "\"albedo_floor\": 0.03"),
                          "inputs.albedo_floor is not the 0.02 this renderer runs with"));
    CHECK(tests::contains(network_refusal("   2,\n   4\n", "   2,\n   8\n"),
                          "its spacings are not contract 18's 1, 2 and 4"));
    CHECK(tests::contains(network_refusal("    24,\n    32\n   ],", "    24\n   ],"),
                          "widths.enc and widths.dec must have 4 entries each"));
    CHECK(tests::contains(network_refusal("\"masked_tap_score\": -10000.0", "\"masked_tap_score\": -1000.0"),
                          "masked_tap_score"));
    CHECK(tests::contains(network_refusal("\"bottleneck\": 40", "\"bottleneck\": 41"), "multiple of 8"));
    CHECK(tests::contains(network_refusal("\"unet.head.bias\"", "\"unet.head.extra\""), "no tensor unet.head.bias"));
    CHECK(tests::contains(network_refusal("{", "{", 1000), "inside weights.bin"));
    CHECK(tests::contains(network_refusal("{", "{ ,"), "layout.json"));
    CHECK(network_refusal("{", "{").empty());
}

TEST_CASE("an export whose file cannot be read whole is refused as unreadable, not as a short file") {
    // A directory where a file should be opens on macOS and reads nothing:
    // read through a stream iterator it was an empty weights.bin, refused
    // later for its size, or an empty layout.json, for its JSON.
    const tests::ScratchDirectory scratch("denoise-unreadable");
    std::filesystem::copy_file(fixture_network / "layout.json", scratch / "layout.json");
    std::filesystem::create_directory(scratch / "weights.bin");
    CHECK(tests::contains(tests::error_of<denoise::NetworkError>([&] { return denoise::load_network(scratch.path()); }),
                          "cannot read weights.bin"));
    std::filesystem::remove(scratch / "layout.json");
    std::filesystem::create_directory(scratch / "layout.json");
    CHECK(tests::contains(tests::error_of<denoise::NetworkError>([&] { return denoise::load_network(scratch.path()); }),
                          "cannot read layout.json"));
    CHECK(tests::contains(
        tests::error_of<denoise::NetworkError>([&] { return denoise::load_network(scratch / "missing"); }),
        "cannot read layout.json"));
}

TEST_CASE("a denoised graph applies its lens after the network, by depth_of_field directly after the denoise pass") {
    CHECK(frame::reads_radiance(PassKind::depth_of_field));
    CHECK_FALSE(frame::writes_radiance(PassKind::depth_of_field));
    CHECK_FALSE(frame::writes_target(PassKind::depth_of_field));
    CHECK(frame::needs_scene(PassKind::depth_of_field));
    CHECK(frame::lens_after_network(schedule_of(denoised)));
    // The network without the lens after it, which would drop the lens.
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade,
                                               PassKind::features, PassKind::denoise, PassKind::tone_map})),
                          "followed directly by depth_of_field"));
    // The lens after anything but the network.
    Schedule path{{PassKind::path, PassKind::depth_of_field, PassKind::tone_map}, tests::look};
    CHECK(tests::contains(refusal(path), "directly after the denoise pass"));
    CHECK(tests::contains(refusal(schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade,
                                               PassKind::features, PassKind::depth_of_field, PassKind::denoise,
                                               PassKind::depth_of_field, PassKind::tone_map})),
                          "depth_of_field more than once"));
}

TEST_CASE("a graph whose light is the network's input traces through its camera's pinhole, and no other does") {
    contracts::Camera lensed{};
    lensed.position = {0.0f, 0.0f, 0.0f};
    lensed.look_at = {0.0f, 0.0f, -1.0f};
    lensed.up = {0.0f, 1.0f, 0.0f};
    lensed.vertical_fov_degrees = 30.0f;
    lensed.lens_radius = 0.01f;
    lensed.focus_distance = 2.0f;
    contracts::Camera before = lensed;
    before.position = {0.1f, 0.0f, 0.0f};
    frame::FrameInputs inputs{.time = frame::Seconds(1.0), .index = 7, .accumulated_since = 7, .camera = lensed};
    inputs.previous = frame::PreviousFrame{frame::Seconds(0.9), before};

    const frame::FrameInputs traced = frame::traced_inputs(schedule_of(denoised), inputs);
    CHECK(traced.camera->lens_radius == 0.0f);
    CHECK(traced.previous->camera.lens_radius == 0.0f);
    // The rest of each camera, and of the frame, as given.
    CHECK(traced.camera->focus_distance == lensed.focus_distance);
    CHECK(traced.camera->vertical_fov_degrees == lensed.vertical_fov_degrees);
    CHECK(traced.previous->camera.position.x == before.position.x);
    CHECK(traced.index == inputs.index);
    CHECK(traced.time == inputs.time);

    // The data set's input graph, without the network, traces as the
    // denoised graph does (logical-overview.md, principle 13).
    const Schedule input = schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::features,
                                        PassKind::tone_map},
                                       false);
    CHECK(frame::traced_inputs(input, inputs).camera->lens_radius == 0.0f);

    const Schedule plain = schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade, PassKind::tone_map},
                                       false);
    const frame::FrameInputs kept = frame::traced_inputs(plain, inputs);
    CHECK(kept.camera->lens_radius == lensed.lens_radius);
    CHECK(kept.previous->camera.lens_radius == before.lens_radius);
}
