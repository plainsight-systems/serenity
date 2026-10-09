// Shapes that move (core/animation/animate.h), on the GPU, end to end: a
// moving scene at time t renders exactly as the same scene still, its shapes
// placed by hand where their motions put them at t, with frames in flight
// in both slots.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/animate.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

frame::Schedule graph(const char* passes) {
    return frame::parse_schedule(std::string("passes = [") + passes + "]\n", "test");
}

std::string number(float v) {
    char text[32];
    std::snprintf(text, sizeof text, "%.9g", static_cast<double>(v));
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
            const contracts::Float3 p = animation::position(placed_from->animation.motions, m.motion,
                                                            frame::Seconds(time));
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

}  // namespace

TEST_CASE("a moving scene at time t renders exactly as the scene still, placed where its motions put it at t") {
    const scene::SceneDescription moving = scene::parse(brass(), "moving");
    REQUIRE(animation::moves(moving.animation));
    const frame::Extent size{96, 64};
    const std::vector<double> times = {0.0, 0.7, 1.9, 4.2, 37.5};

    // Every frame of the moving scene recorded before any is waited for, so
    // both slots' transforms and structures are in flight together.
    std::vector<std::vector<std::uint8_t>> animated;
    {
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, graph("\"preview\""), &moving);
        std::vector<std::unique_ptr<metal::Offscreen>> targets;
        std::uint64_t sequence = 0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
            sequence = metal::render_to_offscreen(
                submission, *targets.back(), renderer,
                frame::FrameInputs{.time = frame::Seconds(times[i]), .index = i, .accumulated_since = i,
                                   .camera = moving.camera});
        }
        (void)submission.wait_until_complete(sequence);
        for (const auto& target : targets) {
            animated.emplace_back(std::size_t{size.width} * size.height * 4);
            target->read_rgba(animated.back());
        }
    }

    for (std::size_t i = 0; i < times.size(); ++i) {
        const scene::SceneDescription still = scene::parse(brass(&moving, times[i]), "still");
        REQUIRE_FALSE(animation::moves(still.animation));
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, graph("\"preview\""), &still);
        metal::Offscreen target(device, submission, size);
        const std::uint64_t sequence = metal::render_to_offscreen(
            submission, target, renderer,
            frame::FrameInputs{.time = frame::Seconds(0.0), .index = 0, .accumulated_since = 0,
                               .camera = still.camera});
        (void)submission.wait_until_complete(sequence);
        std::vector<std::uint8_t> expected(std::size_t{size.width} * size.height * 4);
        target.read_rgba(expected);
        INFO("time " << times[i]);
        CHECK(animated[i] == expected);
    }
    // And the fireflies did move: the first and last frames differ.
    CHECK(animated.front() != animated.back());
}
