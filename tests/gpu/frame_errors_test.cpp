// What a frame's inputs, a scene or a target can get wrong is refused by
// Error before the frame's submission begins (metal/frame/renderer.h,
// prepare), so the run can go on, or end, with nothing left open; and a
// refusal changes nothing (E.4). And every refusal the backend's memory
// owners state: an image Metal cannot make or read back at another size, an
// array of no bytes, a slot or an array index that is not one.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
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
#include "test_paths.h"

using namespace serenity;

TEST_CASE("a frame with no camera is refused before its submission begins, and the run goes on") {
    const scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, tests::path_graph(), &description);
    const std::uint64_t next = submission.next_sequence();
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer, tests::frame_at(0, 0, 0.0, std::nullopt)),
                    metal::MetalError);
    CHECK(submission.next_sequence() == next);  // nothing was begun
    const std::uint64_t done =
        metal::render_to_offscreen(submission, target, renderer, tests::frame_at(0, 0, 0.0, description.camera));
    CHECK_NOTHROW((void)submission.wait_until_complete(done));
}

TEST_CASE("a glowing light that is not one of the scene's is refused when the renderer is built") {
    scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere_flight.toml");
    REQUIRE_FALSE(description.animation.glowers.empty());
    description.animation.glowers.back().target = description.light_counts.spheres;
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(metal::Renderer(device, submission, tests::path_graph(), &description), metal::MetalError);
}

TEST_CASE("an image Metal cannot make is refused by Error, not by Metal stopping the program") {
    metal::Device device;
    metal::Submission submission(device);
    const frame::Extent too_wide{metal::max_texture_side + 1, 16};
    CHECK_THROWS_AS(metal::Offscreen(device, submission, too_wide), metal::MetalError);
    CHECK_THROWS_AS(metal::Offscreen(device, submission, {0, 16}), metal::MetalError);
    CHECK_THROWS_AS(metal::Offscreen(device, submission, {16, 0}), metal::MetalError);
    CHECK_NOTHROW(metal::Offscreen(device, submission, {metal::max_texture_side, 1}));
}

TEST_CASE("an image is read back only into exactly its bytes") {
    metal::Device device;
    metal::Submission submission(device);
    const metal::Offscreen target(device, submission, {8, 4});
    REQUIRE(target.rgba_size() == std::size_t{8} * 4 * 4);
    std::vector<std::uint8_t> short_by_one(target.rgba_size() - 1);
    std::vector<std::uint8_t> long_by_one(target.rgba_size() + 1);
    CHECK_THROWS_AS(target.read_rgba(short_by_one), metal::MetalError);
    CHECK_THROWS_AS(target.read_rgba(long_by_one), metal::MetalError);
    std::vector<std::uint8_t> exact(target.rgba_size());
    CHECK_NOTHROW(target.read_rgba(exact));
}

TEST_CASE("the accumulated image refused at a size it cannot have keeps what it held") {
    const scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Accumulation accumulation(device, submission);
    CHECK(accumulation.prepare(tests::frame_at(0, 0, 0.0, description.camera), {16, 16}, false) == 0);
    MTL::Texture* image = accumulation.texture();
    REQUIRE(image != nullptr);
    // Starting over at a size Metal makes no image of: refused...
    CHECK_THROWS_AS(accumulation.prepare(tests::frame_at(1, 1, 0.0, description.camera),
                                         {metal::max_texture_side + 1, 16}, false),
                    metal::MetalError);
    // ...and nothing changed: the image and what it holds are still frame 0's.
    CHECK(accumulation.texture() == image);
    CHECK(accumulation.prepare(tests::frame_at(1, 0, 0.0, description.camera), {16, 16}, false) == 1);
    CHECK(accumulation.texture() == image);
}

TEST_CASE("the window's drawable must be the size the layer was given") {
    metal::Device device;
    metal::Submission submission(device);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CA::MetalLayer* layer = CA::MetalLayer::layer();
    REQUIRE(layer != nullptr);
    CHECK_THROWS_AS(metal::Presenter(device, submission, metal::LayerHandle{layer}, {0, 0}), metal::MetalError);

    metal::Presenter presenter(device, submission, metal::LayerHandle{layer}, {16, 16});
    metal::Renderer renderer(device, submission, frame::parse_schedule("passes = [\"test_pattern\"]\n", "test"),
                             nullptr);
    const frame::FrameInputs inputs = tests::frame_at(0, 0, 0.0, std::nullopt);
    REQUIRE(metal::render_to_window(submission, presenter, renderer, inputs).has_value());
    // The layer's drawables changed size behind the presenter's back.
    layer->setDrawableSize(CGSize{32.0, 32.0});
    const std::uint64_t next = submission.next_sequence();
    CHECK_THROWS_AS(metal::render_to_window(submission, presenter, renderer, inputs), metal::MetalError);
    CHECK(submission.next_sequence() == next);  // refused before anything was begun
    CHECK_NOTHROW((void)submission.finish());
}

TEST_CASE("an array of no bytes, a frame slot or an array that is not one, is refused") {
    metal::Device device;
    metal::Submission submission(device);
    const std::vector<std::byte> bytes(16);
    for (const auto copies : {metal::FrameArray::Copies::one, metal::FrameArray::Copies::per_frame}) {
        CHECK_THROWS_AS(metal::FrameArray(device, submission, std::span<const std::byte>(), copies), metal::MetalError);
        metal::FrameArray array(device, submission, bytes, copies);
        CHECK_NOTHROW((void)array.bytes(metal::frames_in_flight - 1));
        CHECK_THROWS_AS((void)array.bytes(metal::frames_in_flight), metal::MetalError);
        CHECK_THROWS_AS((void)array.address(metal::frames_in_flight), metal::MetalError);
    }
    metal::NonFinite counters(device, submission);
    CHECK_THROWS_AS((void)counters.address(metal::frames_in_flight), metal::MetalError);
    const std::span<const std::byte> one = bytes;
    const metal::StaticArrays arrays(device, submission, std::span(&one, 1));
    CHECK_NOTHROW((void)arrays.address(0));
    CHECK_THROWS_AS((void)arrays.address(1), metal::MetalError);
    // No arrays at all: there is still the zeroed block, and no array 0.
    const metal::StaticArrays none(device, submission, std::span<const std::span<const std::byte>>());
    CHECK_THROWS_AS((void)none.address(0), metal::MetalError);
}
