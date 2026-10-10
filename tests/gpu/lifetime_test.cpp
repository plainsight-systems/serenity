// What the GPU uses is not freed while it uses it (metal/device/submission.h,
// Lifetime): an exception that ends a run with frames in flight destroys the
// renderer and its target first, and each waits for the GPU before anything
// of its own is released. Run under Metal's validation layer
// (MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1), a use of freed memory is a
// GPU fault that settling reports.
//
// The frames are the path tracer's over the marbles at 512 x 512, long
// enough that, without the wait, they are still running when the objects
// that own their memory are destroyed.

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{512, 512};

frame::FrameInputs inputs(const scene::SceneDescription& scene, std::uint64_t index) {
    return frame::FrameInputs{.time = frame::Seconds{0.0}, .index = index, .accumulated_since = 0,
                              .camera = scene.camera};
}

}  // namespace

TEST_CASE("an exception with frames in flight: the renderer and its target wait for the GPU before releasing") {
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/marbles.toml");
    const frame::Schedule schedule = frame::load_schedule(SERENITY_GRAPHS_DIR "/path.toml");
    metal::Device device;
    metal::Submission submission(device);
    std::uint64_t last = 0;
    const auto run = [&] {
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, schedule, &scene);
        for (std::uint64_t index = 0; index < 2; ++index) {
            last = metal::render_to_offscreen(submission, target, renderer, inputs(scene, index));
        }
        throw std::runtime_error("a failure mid-run, frames in flight");
    };
    CHECK_THROWS_AS(run(), std::runtime_error);
    // Both were destroyed by the throw; neither could have let its memory go
    // before the frames using it had completed.
    CHECK(submission.has_completed(last));
    // And the frames ran on memory that was still theirs: no GPU fault.
    CHECK_NOTHROW(submission.finish());
}

TEST_CASE("memory released while a submission is open: that submission can no longer be committed") {
    metal::Device device;
    metal::Submission submission(device);
    std::optional<metal::Offscreen> target;
    target.emplace(device, submission, frame::Extent{8, 8});
    (void)submission.begin();
    target.reset();  // its texture leaves the residency set
    CHECK_THROWS_AS(submission.commit(), metal::MetalError);
}

TEST_CASE("wait_idle() returns once every committed submission has completed") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK(submission.wait_idle());  // nothing committed
    (void)submission.begin();
    submission.commit();
    (void)submission.begin();  // open: not waited for
    CHECK(submission.wait_idle());
    CHECK(submission.has_completed(0));
    submission.commit();
    CHECK(submission.wait_idle());
    CHECK(submission.has_completed(1));
}

TEST_CASE("the window's presenter takes its layer's residency set off the queue only after the GPU is done") {
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/marbles.toml");
    const frame::Schedule schedule = frame::load_schedule(SERENITY_GRAPHS_DIR "/path.toml");
    metal::Device device;
    metal::Submission submission(device);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    // A layer of no window: Core Animation hands out its drawables all the same.
    CA::MetalLayer* layer = CA::MetalLayer::layer();
    REQUIRE(layer != nullptr);
    std::optional<std::uint64_t> last;
    {
        metal::Presenter presenter(device, submission, metal::LayerHandle{layer}, size);
        metal::Renderer renderer(device, submission, schedule, &scene);
        if (const auto frame = metal::render_to_window(submission, presenter, renderer, inputs(scene, 0))) {
            last = frame->sequence;
        }
    }
    REQUIRE(last.has_value());
    CHECK(submission.has_completed(*last));
    CHECK_NOTHROW(submission.finish());
}
