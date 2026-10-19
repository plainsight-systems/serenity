// What the GPU uses is not freed while it uses it (metal/device/submission.h,
// Lifetime): an exception that ends a run with frames in flight destroys the
// renderer and its target first, and each waits for the GPU before anything
// of its own is released. Run under Metal's validation layer
// (MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1), a use of freed memory is a
// GPU fault that settling reports.
//
// The frames are the path tracer's inside a white cavity at 512 x 512,
// whose paths bounce long, so that, without the wait, they are still
// running when the objects that own their memory are destroyed:
// milliseconds of GPU work against the microseconds between the last
// commit and the throw. Whether they were is checked, not assumed: a GPU
// that finished them first would leave the wait unexercised, and the test
// says so rather than passing. Both are integration tests; the submission's
// protocol (abandoning a submission open when memory is released, what an
// idle wait waits for) is a unit apart from Metal (frame_sequence_test.cpp).

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/scene_text.h"

namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

namespace {

constexpr frame::Extent size{512, 512};

// Inside a white sphere that keeps nine tenths of what it receives, a small
// light at its middle, the camera between: every path bounces some ten
// times before the roulette ends it.
std::string white_cavity() {
  return tests::camera_text("[0, 0, 0.5]", "[0, 0, 1]", 60) +
         tests::sky("[0, 0, 0]", "[0, 0, 0]") +
         "[materials.wall]\nkind = \"rough\"\ncolor = [0.9, 0.9, 0.9]\n"
         "[materials.glow]\nkind = \"emissive\"\nradiance = [20, 20, 20]\n"
         "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = "
         "1\nmaterial = \"wall\"\n"
         "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = "
         "0.05\nmaterial = \"glow\"\n";
}

frame::FrameInputs inputs(const scene::SceneDescription& description,
                          std::uint64_t index) {
  return frame::FrameInputs{.time = frame::Seconds{0.0},
                            .index = index,
                            .accumulated_since = 0,
                            .camera = description.camera};
}

}  // namespace

TEST_CASE(
    "an exception with frames in flight: the renderer and its target wait for "
    "the GPU before releasing") {
  const scene::SceneDescription description =
      scene::parse(white_cavity(), "white cavity");
  const frame::Schedule schedule = tests::path_graph();
  metal::Device device;
  metal::Submission submission(device);
  std::uint64_t last = 0;
  bool in_flight = false;
  const auto run = [&] {
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, schedule, &description);
    for (std::uint64_t index = 0; index < 2; ++index) {
      last = metal::render_to_offscreen(submission, target, renderer,
                                        inputs(description, index));
    }
    in_flight = !submission.has_completed(last);
    throw std::runtime_error("a failure mid-run, frames in flight");
  };
  CHECK_THROWS_AS(run(), std::runtime_error);
  // The frames were still running at the throw, so the destructors had to
  // wait: else nothing here was tested (lengthen the frames).
  CHECK(in_flight);
  // Both were destroyed by the throw; neither could have let its memory go
  // before the frames using it had completed.
  CHECK(submission.has_completed(last));
  // And the frames ran on memory that was still theirs: no GPU fault.
  CHECK_NOTHROW(submission.finish());
}

TEST_CASE(
    "the window's presenter takes its layer's residency set off the queue only "
    "after the GPU is done") {
  const scene::SceneDescription description =
      scene::parse(white_cavity(), "white cavity");
  const frame::Schedule schedule = tests::path_graph();
  metal::Device device;
  metal::Submission submission(device);
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // A layer of no window: Core Animation hands out its drawables all the same.
  CA::MetalLayer* layer = CA::MetalLayer::layer();
  REQUIRE(layer != nullptr);
  std::optional<std::uint64_t> last;
  bool in_flight = false;
  {
    metal::Presenter presenter(device, submission, metal::LayerHandle{layer},
                               size);
    metal::Renderer renderer(device, submission, schedule, &description);
    last =
        metal::render_to_window(submission, presenter, renderer,
                                inputs(description, 0), layer->nextDrawable())
            .sequence;
    in_flight = !submission.has_completed(*last);
  }
  REQUIRE(last.has_value());
  // The frame was still running when the presenter went, so its destructor
  // had to wait: else nothing here was tested (lengthen the frame).
  CHECK(in_flight);
  CHECK(submission.has_completed(*last));
  CHECK_NOTHROW(submission.finish());
}
