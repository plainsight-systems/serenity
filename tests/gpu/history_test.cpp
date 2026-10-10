// The accumulated image's history (core/frame/history.h) as the renderer
// keeps it, on the GPU: of a moving scene it holds one instant, and a frame
// of another is refused; of a still scene, any. The rule itself is tested
// without a GPU (tests/history_test.cpp).

#include <cstdint>
#include <string>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

// A floor under one firefly, which wanders, or, with `moves` false, hangs.
std::string scene_text(bool moves) {
    return std::string(R"(
[camera]
position = [0, 1.5, 3]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.floor]
kind = "rough"
color = [0.8, 0.8, 0.8]
[materials.glow]
kind = "emissive"
radiance = [40, 36, 12]
[[shapes]]
kind = "box"
min = [-20, -0.1, -20]
max = [20, 0, 20]
material = "floor"
[[shapes]]
kind = "sphere"
center = [0, 1, 0]
radius = 0.05
material = "glow"
)") + (moves ? "motion = { kind = \"wander\", reach = 0.25, speed = 0.3, seed = 3 }\n" : "");
}

frame::FrameInputs at(const scene::SceneDescription& s, std::uint64_t index, std::uint64_t since, double time) {
    return frame::FrameInputs{.time = frame::Seconds(time), .index = index, .accumulated_since = since,
                              .camera = s.camera};
}

}  // namespace

TEST_CASE("a moving scene's image holds one instant; a still scene's, any") {
    const frame::Schedule path = frame::parse_schedule("passes = [\"path\", \"display\"]\n", "test");
    const frame::Extent size{32, 24};
    {
        const scene::SceneDescription moving = scene::parse(scene_text(true), "moving");
        REQUIRE(scene::changes(moving));
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, path, &moving);
        metal::Offscreen target(device, submission, size);
        (void)metal::render_to_offscreen(submission, target, renderer, at(moving, 0, 0, 1.5));
        // Another sample of the same instant joins it.
        (void)metal::render_to_offscreen(submission, target, renderer, at(moving, 1, 0, 1.5));
        // A frame of another instant is refused, not averaged in.
        CHECK_THROWS_WITH_AS(metal::render_to_offscreen(submission, target, renderer, at(moving, 2, 0, 1.6)),
                             doctest::Contains("an image holds one instant"), metal::MetalError);
        // Starting over at the new instant is fine.
        const std::uint64_t sequence = metal::render_to_offscreen(submission, target, renderer, at(moving, 2, 2, 1.6));
        (void)submission.wait_until_complete(sequence);
        CHECK(renderer.non_finite_samples() == 0);
    }
    {
        const scene::SceneDescription still = scene::parse(scene_text(false), "still");
        REQUIRE_FALSE(scene::changes(still));
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, path, &still);
        metal::Offscreen target(device, submission, size);
        (void)metal::render_to_offscreen(submission, target, renderer, at(still, 0, 0, 1.5));
        const std::uint64_t sequence = metal::render_to_offscreen(submission, target, renderer, at(still, 1, 0, 9.0));
        (void)submission.wait_until_complete(sequence);
        CHECK(renderer.non_finite_samples() == 0);
    }
}
