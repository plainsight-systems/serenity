// The denoiser's features (contract 17, core/contracts/features.h) beside
// a light graph, an integration test: the light of a graph with the
// features pass is the graph's without it, to the bit, as the training
// data's light must be (graphs/restir_pt_features.toml renders it beside
// its features). ReSTIR PT, its reuse running across three frames of
// wandering fireflies, in graphs this test builds.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/same_bits.h"
#include "support/scene_text.h"
#include "support/schedules.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{64, 48};

// ReSTIR PT's whole graph, the features pass beside it where `features`
// says, shown through the tone map, its settings the written ones.
frame::Schedule light_graph(bool features) {
    std::vector<frame::PassKind> kinds = tests::restir_pt_passes;
    if (features) {
        kinds.push_back(frame::PassKind::features);
    }
    kinds.push_back(frame::PassKind::tone_map);
    frame::Schedule s{kinds, tests::look};
    s.restir_pt = tests::written_restir_pt(kinds);
    return s;
}

}  // namespace

TEST_CASE("features: a graph's light is the same to the bit with the features pass beside it") {
    const scene::SceneDescription description = scene::parse(tests::wandering_fireflies(), "wandering fireflies");
    const auto accumulated = [&](const frame::Schedule& graph) {
        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, graph, &description);
        std::uint64_t last = 0;
        for (std::uint64_t i = 0; i < 3; ++i) {
            // Its reuse history kept from frame 0 (reused_since, the default).
            const frame::FrameInputs inputs =
                tests::frame_at(i, i, static_cast<double>(i) / 60.0, description.camera);
            last = metal::render_to_offscreen(submission, target, renderer, inputs);
        }
        (void)submission.wait_until_complete(last);
        CHECK(renderer.non_finite_samples() == 0);
        std::vector<float> image(std::size_t{size.width} * size.height * 4);
        renderer.read_accumulated(image);
        return image;
    };
    const std::vector<float> without = accumulated(light_graph(false));
    const std::vector<float> with = accumulated(light_graph(true));
    CHECK(tests::same_bits(with, without));
    // And it holds light: not two black images alike.
    CHECK(std::ranges::max(with) > 0.0f);
}
