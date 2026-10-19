// What a frame's inputs, a scene or a target can get wrong is refused by
// Error before the frame's submission begins (metal/frame/renderer.h,
// prepare), so the run can go on, or end, with nothing left open; and a
// refusal changes nothing (E.4). And every refusal the backend's memory
// owners state: an image Metal cannot make or read back at another size, an
// array of no bytes, a slot or an array index that is not one. And the
// window's: a drawable not the size it renders at, refused; a display link
// at a rate it cannot ask for, or off the main thread.

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "core/frame/graph_file.h"
#include "core/presentation/frame_source.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/display_link.h"
#include "metal/device/error.h"
#include "metal/device/frame_array.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/film/failure_count.h"
#include "metal/frame/accumulation.h"
#include "metal/frame/renderer.h"
#include "support/scene_text.h"

namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace presentation = serenity::presentation;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

TEST_CASE(
    "a frame with no camera is refused before its submission begins, and the "
    "run goes on") {
  const scene::SceneDescription description =
      scene::parse(tests::still_fireflies(), "still fireflies");
  metal::Device device;
  metal::Submission submission(device);
  metal::Offscreen target(device, submission, {16, 16});
  metal::Renderer renderer(device, submission, tests::path_graph(),
                           &description);
  const std::uint64_t next = submission.next_sequence();
  CHECK_THROWS_AS(
      metal::render_to_offscreen(submission, target, renderer,
                                 tests::frame_at(0, 0, 0.0, std::nullopt)),
      metal::MetalError);
  CHECK(submission.next_sequence() == next);  // nothing was begun
  const std::uint64_t done = metal::render_to_offscreen(
      submission, target, renderer,
      tests::frame_at(0, 0, 0.0, description.camera));
  CHECK_NOTHROW((void)submission.wait_until_complete(done));
}

TEST_CASE(
    "a glowing light that is not one of the scene's is refused when the "
    "renderer is built") {
  scene::SceneDescription description =
      scene::parse(tests::flying_fireflies(), "flying fireflies");
  REQUIRE_FALSE(description.animation.glowers.empty());
  description.animation.glowers.back().target =
      description.light_counts.spheres;
  metal::Device device;
  metal::Submission submission(device);
  CHECK_THROWS_AS(
      metal::Renderer(device, submission, tests::path_graph(), &description),
      metal::MetalError);
}

TEST_CASE(
    "an image Metal cannot make is refused by Error, not by Metal stopping the "
    "program") {
  metal::Device device;
  metal::Submission submission(device);
  const frame::Extent too_wide{metal::max_texture_side + 1, 16};
  CHECK_THROWS_AS(metal::Offscreen(device, submission, too_wide),
                  metal::MetalError);
  CHECK_THROWS_AS(metal::Offscreen(device, submission, {0, 16}),
                  metal::MetalError);
  CHECK_THROWS_AS(metal::Offscreen(device, submission, {16, 0}),
                  metal::MetalError);
  CHECK_NOTHROW(
      metal::Offscreen(device, submission, {metal::max_texture_side, 1}));
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

TEST_CASE(
    "the accumulated image refused at a size it cannot have keeps what it "
    "held") {
  const scene::SceneDescription description =
      scene::parse(tests::still_fireflies(), "still fireflies");
  metal::Device device;
  metal::Submission submission(device);
  metal::Accumulation accumulation(device, submission);
  CHECK(accumulation.prepare(tests::frame_at(0, 0, 0.0, description.camera),
                             {16, 16}, frame::SceneMotion::still) == 0);
  MTL::Texture* image = accumulation.texture();
  REQUIRE(image != nullptr);
  // Starting over at a size Metal makes no image of: refused...
  CHECK_THROWS_AS(
      accumulation.prepare(tests::frame_at(1, 1, 0.0, description.camera),
                           {metal::max_texture_side + 1, 16},
                           frame::SceneMotion::still),
      metal::MetalError);
  // ...and nothing changed: the image and what it holds are still frame 0's.
  CHECK(accumulation.texture() == image);
  CHECK(accumulation.prepare(tests::frame_at(1, 0, 0.0, description.camera),
                             {16, 16}, frame::SceneMotion::still) == 1);
  CHECK(accumulation.texture() == image);
}

TEST_CASE(
    "a window's drawable must be the presenter's size, and the presenter "
    "restores the layer's size") {
  metal::Device device;
  metal::Submission submission(device);
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  CA::MetalLayer* layer = CA::MetalLayer::layer();
  REQUIRE(layer != nullptr);
  CHECK_THROWS_AS(
      metal::Presenter(device, submission, metal::LayerHandle{layer}, {0, 0}),
      metal::MetalError);

  metal::Presenter presenter(device, submission, metal::LayerHandle{layer},
                             {16, 16});
  metal::Renderer renderer(
      device, submission,
      frame::parse_schedule("passes = [\"test_pattern\"]\n", "test"), nullptr);
  const frame::FrameInputs inputs = tests::frame_at(0, 0, 0.0, std::nullopt);
  CA::MetalDrawable* fits = layer->nextDrawable();
  CHECK(presenter.renders_into(fits));
  (void)metal::render_to_window(submission, presenter, renderer, inputs, fits);
  CHECK_FALSE(presenter.renders_into(nullptr));
  // The layer's drawables changed size behind the presenter's back, as
  // SDL changes them on a resize: one of them is not rendered into, and
  // refused, before anything is begun (the passes would write past a
  // smaller one)...
  layer->setDrawableSize(CGSize{32.0, 32.0});
  CA::MetalDrawable* stale = layer->nextDrawable();
  CHECK_FALSE(presenter.renders_into(stale));
  const std::uint64_t next = submission.next_sequence();
  CHECK_THROWS_AS((void)metal::render_to_window(submission, presenter, renderer,
                                                inputs, stale),
                  metal::MetalError);
  CHECK_THROWS_AS((void)metal::render_to_window(submission, presenter, renderer,
                                                inputs, nullptr),
                  metal::MetalError);
  CHECK(submission.next_sequence() == next);
  // ...and following the window to the size it has sets the layer's back,
  // so the next drawable is one to render into.
  presenter.resize({16, 16});
  CHECK(layer->drawableSize().width == 16.0);
  CHECK(layer->drawableSize().height == 16.0);
  CHECK(presenter.renders_into(layer->nextDrawable()));
  CHECK_NOTHROW((void)submission.finish());
}

TEST_CASE(
    "a display link is refused a frame rate not positive and in order, and any "
    "thread but the main one") {
  metal::Device device;
  metal::Submission submission(device);
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  CA::MetalLayer* layer = CA::MetalLayer::layer();
  REQUIRE(layer != nullptr);
  const metal::Presenter presenter(device, submission,
                                   metal::LayerHandle{layer}, {16, 16});
  struct Counting final : metal::DisplayLink::Client {
    int frames = 0;
    void frame(const metal::DisplayUpdate&) noexcept override { ++frames; }
  } client;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const presentation::FrameRate wrong :
       {presentation::FrameRate{0.0F, 0.0F, 0.0F},
        presentation::FrameRate{-60.0F, 60.0F, 60.0F},
        presentation::FrameRate{60.0F, 30.0F, 45.0F},
        presentation::FrameRate{30.0F, 60.0F, 90.0F},
        presentation::FrameRate{30.0F, 60.0F, 20.0F},
        presentation::FrameRate{30.0F, 60.0F, nan},
        presentation::FrameRate{30.0F, infinity, 60.0F}}) {
    INFO("rate " << wrong.minimum << " to " << wrong.maximum << ", preferred "
                 << wrong.preferred);
    CHECK_THROWS_AS(metal::DisplayLink(presenter, wrong, client),
                    metal::MetalError);
  }
  CHECK_NOTHROW(metal::DisplayLink(
      presenter, presentation::FrameRate{60.0F, 60.0F, 60.0F}, client));
  // Off the main thread its frames would be called on a run loop nobody
  // runs (display_link.h).
  bool refused = false;
  std::thread([&] {
    try {
      const metal::DisplayLink link(
          presenter, presentation::FrameRate{60.0F, 60.0F, 60.0F}, client);
    } catch (const metal::MetalError&) {
      refused = true;
    }
  }).join();
  CHECK(refused);
  // Frames are called by the run loop alone, which no test runs: never by
  // the link's making or its end.
  CHECK(client.frames == 0);
}

TEST_CASE(
    "an array of no bytes, a frame slot or an array that is not one, is "
    "refused") {
  metal::Device device;
  metal::Submission submission(device);
  const std::vector<std::byte> bytes(16);
  for (const auto copies :
       {metal::FrameArray::Copies::one, metal::FrameArray::Copies::per_frame}) {
    CHECK_THROWS_AS(metal::FrameArray(device, submission,
                                      std::span<const std::byte>(), copies),
                    metal::MetalError);
    metal::FrameArray array(device, submission, bytes, copies);
    CHECK_NOTHROW((void)array.bytes(metal::frames_in_flight - 1));
    CHECK_THROWS_AS((void)array.bytes(metal::frames_in_flight),
                    metal::MetalError);
    CHECK_THROWS_AS((void)array.address(metal::frames_in_flight),
                    metal::MetalError);
  }
  metal::FailureCount counters(device, submission, "samples not finite");
  CHECK_THROWS_AS((void)counters.address(metal::frames_in_flight),
                  metal::MetalError);
  const std::span<const std::byte> one = bytes;
  const metal::StaticArrays arrays(device, submission, std::span(&one, 1));
  CHECK_NOTHROW((void)arrays.address(0));
  CHECK_THROWS_AS((void)arrays.address(1), metal::MetalError);
  // No arrays at all: there is still the zeroed block, and no array 0.
  const metal::StaticArrays none(device, submission,
                                 std::span<const std::span<const std::byte>>());
  CHECK_THROWS_AS((void)none.address(0), metal::MetalError);
}
