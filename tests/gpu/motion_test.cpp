// Shapes that move (core/animation/animate.h), on the GPU, end to end: a
// moving scene at time t renders exactly as the same scene still, its shapes
// placed by hand where their motions put them at t, with frames in flight
// in both slots.

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/animate.h"
#include "core/contracts/transform.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "test_paths.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{96, 64};

// `v` written so that reading it back gives `v` exactly.
std::string number(float v) {
    std::string text(32, '\0');
    const std::to_chars_result written = std::to_chars(text.data(), text.data() + text.size(), v);
    REQUIRE(written.ec == std::errc{});
    text.resize(static_cast<std::size_t>(written.ptr - text.data()));
    return text;
}

// The wandering brass scene, small, with its fireflies' motions; or, given
// a time, the same scene still, each firefly placed where its motion puts
// it at that time (the anchor and the radius as in the file).
std::string brass(const scene::SceneDescription* placed_from = nullptr, double time = 0.0) {
    const auto firefly = [&](const char* center, const char* radius, const char* motion, std::size_t which) {
        std::string at = center;
        std::string moves = std::string("motion = ") + motion + "\n";
        if (placed_from != nullptr) {
            const animation::Mover& m = placed_from->animation.movers.at(which);
            const contracts::Float3 p =
                animation::position(placed_from->animation.motions, m.motion, frame::Seconds(time));
            at = "[" + number(p.x) + ", " + number(p.y) + ", " + number(p.z) + "]";
            moves = "";
        }
        return "[[shapes]]\nkind = \"sphere\"\ncenter = " + at + "\nradius = " + radius +
               "\nmaterial = \"firefly\"\n" + moves;
    };
    return std::string(R"(
[camera]
position = [0.0, 1.1, 4.2]
look_at = [0.0, 0.75, 0.0]
vertical_fov_degrees = 38
[environment]
kind = "gradient"
zenith = [0.002, 0.003, 0.012]
horizon = [0.012, 0.016, 0.035]
[materials.brass]
kind = "conductor"
f0 = [0.91, 0.78, 0.42]
roughness = 0.35
[materials.floor]
kind = "rough"
color = [0.7, 0.68, 0.6]
[materials.firefly]
kind = "emissive"
radiance = [320.0, 280.0, 70.0]
[[shapes]]
kind = "sphere"
center = [0.0, 0.75, 0.0]
radius = 0.75
material = "brass"
[[shapes]]
kind = "box"
min = [-20.0, -0.1, -20.0]
max = [20.0, 0.0, 20.0]
material = "floor"
)") + firefly("[-1.05, 0.55, 0.85]", "0.035", "{ kind = \"wander\", reach = 0.25, speed = 0.25, seed = 1 }", 0) +
           firefly("[0.95, 1.45, -0.55]", "0.03", "{ kind = \"wander\", reach = 0.25, speed = 0.3, seed = 2 }", 1);
}

// The preview of `description` at each of `times`, every frame recorded
// before any is waited for, so both slots' transforms and structures are in
// flight together.
std::vector<std::vector<std::uint8_t>> animated(const scene::SceneDescription& description,
                                                const std::vector<double>& times) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, tests::preview_graph(), &description);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    std::uint64_t sequence = 0;
    for (std::uint64_t i = 0; i < times.size(); ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
        sequence = metal::render_to_offscreen(submission, *targets.back(), renderer,
                                              tests::frame_at(i, i, times[i], description.camera));
    }
    (void)submission.wait_until_complete(sequence);
    std::vector<std::vector<std::uint8_t>> images;
    for (const auto& target : targets) {
        images.push_back(tests::read_back(*target));
    }
    return images;
}

// `description` held still: rendered alone at time 0.
std::vector<std::uint8_t> still(const scene::SceneDescription& description) {
    return tests::render_once(description, tests::preview_graph(), size,
                              tests::frame_at(0, 0, 0.0, description.camera));
}

}  // namespace

TEST_CASE("a moving scene at time t renders exactly as the scene still, placed where its motions put it at t") {
    const scene::SceneDescription moving = scene::parse(brass(), "moving");
    REQUIRE(animation::moves(moving.animation));
    const std::vector<double> times = {0.0, 0.7, 1.9, 4.2, 37.5};
    const std::vector<std::vector<std::uint8_t>> frames = animated(moving, times);

    for (std::size_t i = 0; i < times.size(); ++i) {
        const scene::SceneDescription placed = scene::parse(brass(&moving, times[i]), "still");
        REQUIRE_FALSE(animation::moves(placed.animation));
        INFO("time " << times[i]);
        CHECK(tests::differing(frames[i], still(placed)) == 0);
    }
    // And the fireflies did move: the first and last frames differ.
    CHECK(tests::differing(frames.front(), frames.back()) > 0);
}

TEST_CASE("a flying, blinking scene at time t renders exactly as the scene still, placed and lit as at t") {
    const scene::SceneDescription flying = scene::load(tests::scenes_dir / "brass_sphere_flight.toml");
    REQUIRE(animation::moves(flying.animation));
    REQUIRE(flying.animation.glowers.size() == 6);
    const std::vector<double> times = {0.0, 2.6, 13.1, 77.7};
    const std::vector<std::vector<std::uint8_t>> frames = animated(flying, times);

    bool some_dim = false;
    for (std::size_t i = 0; i < times.size(); ++i) {
        // The same scene, nothing moving or glowing: each firefly placed
        // where its flight is at t, its radiance scaled by its glow at t.
        scene::SceneDescription placed = flying;
        placed.animation = {};
        const frame::Seconds t(times[i]);
        for (const animation::Mover& m : flying.animation.movers) {
            placed.shapes.transforms[m.target] = contracts::moved_to(
                placed.shapes.transforms[m.target], animation::position(flying.animation.motions, m.motion, t));
        }
        for (const animation::Glower& g : flying.animation.glowers) {
            const float factor = animation::glow(flying.animation.glows, g.glow, t);
            some_dim = some_dim || factor < 0.5f;
            contracts::Float3& radiance = placed.sphere_lights[g.target].radiance;
            radiance = {radiance.x * factor, radiance.y * factor, radiance.z * factor};
        }
        INFO("time " << times[i]);
        CHECK(tests::differing(frames[i], still(placed)) == 0);
    }
    // The glows were in play: some firefly was dim at some time.
    CHECK(some_dim);
}
