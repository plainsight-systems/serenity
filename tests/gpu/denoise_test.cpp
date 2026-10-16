// The denoise pass (passes/denoise/denoise.h) against PyTorch: the
// committed fixture, tests/data/denoise (training/denoise_check.py made it:
// a network of training/model.py's structure at small widths, randomly
// initialized, head included; three consecutive frames of a training
// sequence's renderer images, a 40 x 22 crop where fireflies move, so the
// U-Net's padding, the masked taps, the depth median, the motion vectors
// and the history all take part; and model.Denoiser.frame's float32
// outputs). Run through the GPU as the renderer runs it, frame after frame
// with its own history:
//
//   - each output is PyTorch's to within `tolerance`, relative: half
//     floats in the U-Net (denoise.metal, "Precision"), float sums. On this
//     fixture the largest is 2.2e-4 (docs/research/2026-10-15-denoiser-
//     live.md), so 1e-3 leaves a margin of four;
//   - a network one weight off is not: one of the last convolution's
//     weights a tenth off gives 4.0e-3, four times the tolerance.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "core/frame/graph_file.h"
#include "core/output/pfm.h"
#include "core/scene/scene.h"
#include "gpu/support/denoise_fixture.h"
#include "gpu/support/rendering.h"
#include "metal/device/offscreen.h"
#include "metal/frame/renderer.h"
#include "test_paths.h"

using namespace serenity;

namespace {

constexpr double tolerance = 1e-3;
// Below this, a value counts as this: the floor of the relative test, so
// near-black pixels are held to an absolute 1e-6.
constexpr double floor_value = 1e-3;

// The largest |gpu - expected| / (|expected| + floor_value), over every
// frame, pixel and channel.
double worst(const tests::DenoiseFixture& fixture, const std::vector<contracts::LinearImage>& outputs) {
    REQUIRE(outputs.size() == fixture.frames);
    double largest = 0.0;
    for (std::size_t k = 0; k < outputs.size(); ++k) {
        const contracts::LinearImage expected = output::read_pfm(fixture.file(k, "expected"));
        REQUIRE(expected.extent == outputs[k].extent);
        for (std::size_t i = 0; i < expected.rgb.size(); ++i) {
            const double e = expected.rgb[i];
            const double g = outputs[k].rgb[i];
            REQUIRE(std::isfinite(g));
            largest = std::max(largest, std::abs(g - e) / (std::abs(e) + floor_value));
        }
    }
    return largest;
}

}  // namespace

TEST_CASE("the denoise pass computes what PyTorch computes, frame after frame") {
    const tests::DenoiseFixture fixture = tests::read_denoise_fixture(tests::data_dir / "denoise");
    REQUIRE(fixture.frames == 3);
    const double error = worst(fixture, tests::run_denoise_fixture(fixture));
    INFO("largest relative difference: " << error);
    CHECK(error < tolerance);
}

TEST_CASE("a network one weight off does not pass the denoise check") {
    const tests::DenoiseFixture fixture = tests::read_denoise_fixture(tests::data_dir / "denoise");
    // One weight of the last convolution, by a tenth: about the size of
    // its weights.
    const double error = worst(fixture, tests::run_denoise_fixture(fixture, [](denoise::Network& network) {
        network.dec_final.weight.at(0) += 0.1f;
    }));
    INFO("largest relative difference: " << error);
    CHECK(error > tolerance);
}

// The denoised graph in the renderer: graphs/restir_pt_denoised.toml with
// the fixture's network, over the brass sphere scene at 64 x 48 (a 32 x 24
// light). The denoiser's history starts over with ReSTIR's (frame::
// ReuseHistory, metal/frame/denoise_images.h): a frame whose history
// starts over, after frames that came before it or at another size, is
// the frame a new renderer renders, to the bit; one that keeps its
// history is not.
namespace {

frame::Schedule denoised_graph() {
    std::ifstream file{tests::graphs_dir / "restir_pt_denoised.toml"};
    std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const std::string from = "\"../../serenity-data/denoiser-v1-model/wide\"";
    const std::size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    text.replace(at, from.size(), "\"" + (tests::data_dir / "denoise" / "network").string() + "\"");
    return frame::parse_schedule(text, "restir_pt_denoised.toml, the fixture's network");
}

frame::FrameInputs frame_of(std::uint64_t index, std::uint64_t since, const scene::SceneDescription& scene) {
    // A changing scene: the accumulated image starts over every frame.
    frame::FrameInputs inputs = tests::frame_at(index, index, 30.0 + static_cast<double>(index) / 60.0, scene.camera);
    inputs.reused_since = since;
    if (index > since) {
        inputs.previous = frame::PreviousFrame{frame::Seconds{30.0 + static_cast<double>(index - 1) / 60.0}, scene.camera};
    }
    return inputs;
}

// The frames `inputs`, each at its size, on one renderer; the last one's
// denoised radiance (Renderer::read_accumulated).
std::vector<float> rendered(const std::vector<std::pair<frame::FrameInputs, frame::Extent>>& frames,
                            const scene::SceneDescription& scene) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, denoised_graph(), &scene);
    std::vector<float> out;
    for (const auto& [inputs, size] : frames) {
        metal::Offscreen target(device, submission, size);
        (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs));
        out.assign(std::size_t{size.width} * size.height * 4, 0.0f);
        renderer.read_accumulated(out);
    }
    return out;
}

}  // namespace

TEST_CASE("the denoised graph's history starts over with ReSTIR's: at a history's first frame and a new size") {
    const scene::SceneDescription scene = scene::load(tests::scenes_dir / "brass_sphere_wander.toml");
    constexpr frame::Extent size{64, 48};
    constexpr frame::Extent other{80, 40};
    const std::vector<float> fresh = rendered({{frame_of(2, 2, scene), size}}, scene);
    // After two frames, a frame that starts its history over.
    CHECK(rendered({{frame_of(0, 0, scene), size}, {frame_of(1, 0, scene), size}, {frame_of(2, 2, scene), size}},
                   scene) == fresh);
    // After a frame at another size, the frame at this one (a resize starts the history over).
    CHECK(rendered({{frame_of(1, 1, scene), other}, {frame_of(2, 2, scene), size}}, scene) == fresh);
    // The same frame with its history: not the fresh one.
    CHECK(rendered({{frame_of(0, 0, scene), size}, {frame_of(1, 0, scene), size}, {frame_of(2, 0, scene), size}},
                   scene) != fresh);
}
