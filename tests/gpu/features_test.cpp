// The denoiser's features (contract 17, core/contracts/features.h) beside
// a light graph, an integration test: the light of a graph with the
// features pass is the graph's without it, to the bit, as the training
// data's light must be (graphs/restir_pt_features.toml renders it beside
// its features). ReSTIR PT, its reuse running across three frames of
// wandering fireflies, in graphs this test builds; and the rays both take,
// through the frame's camera jitter.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/frame/schedule.h"
#include "core/sampling/jitter.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/lit_floor.h"
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

TEST_CASE("a ReSTIR PT frame's camera rays, the features' with them, all pass through the frame's jitter") {
    // The features' depth of the lit floor at each pixel is the floor's
    // depth along the ray through the pixel's center plus the frame's
    // jitter (core/sampling/jitter.h), every pixel the same offset: a
    // pixel's own random point would be up to half a pixel away, and the
    // floor's depth changes some 1% a pixel.
    const scene::SceneDescription description = scene::parse(tests::lit_floor(tests::one_firefly), "lit floor");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, light_graph(true), &description);
    const contracts::CameraData camera = camera::shader_form(description.camera, size);
    const tests::Vec3 origin = tests::vec(camera.origin);
    const tests::Vec3 forward = tests::vec(camera.forward);
    for (std::uint64_t i = 0; i < 3; ++i) {
        const std::uint64_t last = metal::render_to_offscreen(
            submission, target, renderer, tests::frame_at(i, i, static_cast<double>(i) / 60.0, description.camera));
        (void)submission.wait_until_complete(last);
        const contracts::LinearImage depth = renderer.read_features().depth;
        const sampling::CameraJitter jitter = sampling::camera_jitter(i);
        double worst = 0.0;
        std::size_t floor_pixels = 0;
        for (std::uint32_t y = 0; y < size.height; ++y) {
            for (std::uint32_t x = 0; x < size.width; ++x) {
                const float seen = depth.rgb[(std::size_t{y} * size.width + x) * 3];
                if (seen <= 0.0f) {
                    continue;  // the sky
                }
                const tests::Vec3 at = tests::floor_point(description, size, x + 0.5 + jitter.x, y + 0.5 + jitter.y);
                const double expected = tests::dot(at - origin, forward);
                worst = std::max(worst, std::abs(seen - expected) / expected);
                ++floor_pixels;
            }
        }
        INFO("frame " << i << ": jitter (" << jitter.x << ", " << jitter.y << "), worst relative depth error "
                      << worst);
        CHECK(floor_pixels > size.width * size.height / 2);
        CHECK(worst < 1e-5);
    }
}
