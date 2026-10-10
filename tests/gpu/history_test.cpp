// The accumulated image's history (core/frame/history.h) as the renderer
// keeps it, on the GPU: of a moving scene it holds one instant, and a frame
// of another is refused; of a still scene, any. The rule itself is tested
// without a GPU (tests/history_test.cpp).

#include <cstdint>
#include <string>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
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

constexpr frame::Extent size{32, 24};

// The path tracer over a scene, one frame at a time.
class Rig {
public:
    explicit Rig(bool moves) : description_(scene::parse(scene_text(moves), "test scene")) {}

    // Frame `index`, accumulated since `since`, at `time`; its sequence.
    std::uint64_t render(std::uint64_t index, std::uint64_t since, double time) {
        return metal::render_to_offscreen(submission_, target_, renderer_,
                                          tests::frame_at(index, since, time, description_.camera));
    }

    // Waits for submission `sequence`; the samples left out so far.
    std::uint64_t settle(std::uint64_t sequence) {
        (void)submission_.wait_until_complete(sequence);
        return renderer_.non_finite_samples();
    }

    bool changes() const { return scene::changes(description_); }

private:
    scene::SceneDescription description_;
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Renderer renderer_{device_, submission_, tests::path_graph(), &description_};
    metal::Offscreen target_{device_, submission_, size};
};

}  // namespace

TEST_CASE("a moving scene's image holds one instant; a still scene's, any") {
    {
        Rig moving(true);
        REQUIRE(moving.changes());
        (void)moving.render(0, 0, 1.5);
        // Another sample of the same instant joins it.
        (void)moving.render(1, 0, 1.5);
        // A frame of another instant is refused, not averaged in.
        CHECK_THROWS_WITH_AS(moving.render(2, 0, 1.6), doctest::Contains("an image holds one instant"),
                             metal::MetalError);
        // Starting over at the new instant is fine.
        CHECK(moving.settle(moving.render(2, 2, 1.6)) == 0);
    }
    {
        Rig still(false);
        REQUIRE_FALSE(still.changes());
        (void)still.render(0, 0, 1.5);
        // Another time joins it: a still scene looks the same at any.
        CHECK(still.settle(still.render(1, 0, 9.0)) == 0);
    }
}
