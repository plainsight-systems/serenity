// What a frame's inputs, a scene or a target can get wrong is refused by
// Error before the frame's submission begins (metal/frame/renderer.h,
// prepare), so the run can go on, or end, with nothing left open; and a
// refusal changes nothing (E.4).

#include <cstdint>
#include <optional>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/frame_array.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/film/non_finite.h"
#include "metal/frame/accumulation.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

frame::Schedule path_graph() {
    return frame::parse_schedule("passes = [\"path\", \"display\"]\n", "test");
}

frame::FrameInputs at(std::uint64_t index, std::uint64_t since, const scene::SceneDescription& scene) {
    return frame::FrameInputs{.time = frame::Seconds{0.0}, .index = index, .accumulated_since = since,
                              .camera = scene.camera};
}

}  // namespace

TEST_CASE("a frame with no camera is refused before its submission begins, and the run goes on") {
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, path_graph(), &scene);
    const std::uint64_t next = submission.next_sequence();
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer,
                                               frame::FrameInputs{.time = frame::Seconds{0.0}, .index = 0}),
                    metal::Error);
    CHECK(submission.next_sequence() == next);  // nothing was begun
    const std::uint64_t done = metal::render_to_offscreen(submission, target, renderer, at(0, 0, scene));
    CHECK_NOTHROW(submission.wait_until_complete(done));
}

TEST_CASE("a glowing light that is not one of the scene's is refused when the renderer is built") {
    scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere_flight.toml");
    REQUIRE_FALSE(scene.animation.glowers.empty());
    scene.animation.glowers.back().target = scene.light_counts.spheres;
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(metal::Renderer(device, submission, path_graph(), &scene), metal::Error);
}

TEST_CASE("an image Metal cannot make is refused by Error, not by Metal stopping the program") {
    metal::Device device;
    metal::Submission submission(device);
    const frame::Extent too_wide{metal::max_texture_side + 1, 16};
    CHECK_THROWS_AS(metal::Offscreen(device, submission, too_wide), metal::Error);
    CHECK_THROWS_AS(metal::Offscreen(device, submission, {0, 16}), metal::Error);
    CHECK_NOTHROW(metal::Offscreen(device, submission, {metal::max_texture_side, 1}));
}

TEST_CASE("the accumulated image refused at a size it cannot have keeps what it held") {
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Accumulation accumulation(device, submission);
    CHECK(accumulation.prepare(at(0, 0, scene), {16, 16}, false) == 0);
    MTL::Texture* image = accumulation.texture();
    REQUIRE(image != nullptr);
    // Starting over at a size Metal makes no image of: refused...
    CHECK_THROWS_AS(accumulation.prepare(at(1, 1, scene), {metal::max_texture_side + 1, 16}, false), metal::Error);
    // ...and nothing changed: the image and what it holds are still frame 0's.
    CHECK(accumulation.texture() == image);
    CHECK(accumulation.prepare(at(1, 0, scene), {16, 16}, false) == 1);
    CHECK(accumulation.texture() == image);
}

TEST_CASE("the window's drawable must be the size the layer was given") {
    metal::Device device;
    metal::Submission submission(device);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CA::MetalLayer* layer = CA::MetalLayer::layer();
    REQUIRE(layer != nullptr);
    CHECK_THROWS_AS(metal::Presenter(device, submission, metal::LayerHandle{layer}, {0, 0}), metal::Error);

    metal::Presenter presenter(device, submission, metal::LayerHandle{layer}, {16, 16});
    metal::Renderer renderer(device, submission, frame::parse_schedule("passes = [\"test_pattern\"]\n", "test"),
                             nullptr);
    const frame::FrameInputs inputs{.time = frame::Seconds{0.0}, .index = 0};
    REQUIRE(metal::render_to_window(submission, presenter, renderer, inputs).has_value());
    // The layer's drawables changed size behind the presenter's back.
    layer->setDrawableSize(CGSize{32.0, 32.0});
    const std::uint64_t next = submission.next_sequence();
    CHECK_THROWS_AS(metal::render_to_window(submission, presenter, renderer, inputs), metal::Error);
    CHECK(submission.next_sequence() == next);  // refused before anything was begun
    CHECK_NOTHROW(submission.finish());
}

TEST_CASE("a frame slot that is not one is refused, however many copies an array has") {
    metal::Device device;
    metal::Submission submission(device);
    const std::vector<std::byte> bytes(16);
    for (const std::uint32_t copies : {1u, metal::frames_in_flight}) {
        metal::FrameArray array(device, submission, bytes, copies);
        CHECK_NOTHROW(array.bytes(metal::frames_in_flight - 1));
        CHECK_THROWS_AS(array.bytes(metal::frames_in_flight), metal::Error);
        CHECK_THROWS_AS(array.address(metal::frames_in_flight), metal::Error);
    }
    metal::NonFinite counters(device, submission);
    CHECK_THROWS_AS(counters.address(metal::frames_in_flight), metal::Error);
    const std::span<const std::byte> one = bytes;
    metal::StaticArrays arrays(device, submission, std::span(&one, 1));
    CHECK_NOTHROW(arrays.address(0));
    CHECK_THROWS_AS(arrays.address(1), metal::Error);
}
