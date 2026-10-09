// Shapes that move (core/animation/animate.h), on the GPU:
//
//   - the structure: a ray's hit, tested in a shape's own space, is at the
//     world's distance and names the shape, at any scale, and a moved shape
//     is hit where it moved to and not where it was
//     (metal/acceleration/scene_acceleration.h);
//   - a frame: a moving scene at time t renders exactly as the same scene
//     still, its shapes placed by hand where their motions put them at t,
//     with frames in flight in both slots;
//   - the accumulated image: of a moving scene it holds one instant, and a
//     frame of another is refused; of a still scene, any.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/animate.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

namespace {

struct ProbeRay {
    float origin[4];
    float direction[4];
};

struct ProbeHit {
    float t;
    float shape;  // -1 t for none
};

// Traces `rays` through `acceleration`'s structure for `slot`, with the scene's
// shapes as `transforms` places them, in one command buffer; if `moving`,
// first rewrites the slot's transforms and records the structure's update.
std::vector<ProbeHit> probe(metal::Device& device, metal::Submission& submission, const metal::SceneBuffers& buffers, const metal::ShapeTransforms& transforms,
                            const metal::SceneAcceleration& acceleration, const std::vector<ProbeRay>& rays,
                            const std::vector<contracts::Transform>* moved) {
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline("trace_probe");
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const std::uint32_t count = static_cast<std::uint32_t>(rays.size());
    auto out = NS::TransferPtr(mtl->newBuffer(count * sizeof(ProbeHit), MTL::ResourceStorageModeShared));
    auto ray_buffer = NS::TransferPtr(mtl->newBuffer(rays.data(), count * sizeof(ProbeRay),
                                                     MTL::ResourceStorageModeShared));
    auto count_buffer = NS::TransferPtr(mtl->newBuffer(&count, sizeof count, MTL::ResourceStorageModeShared));
    REQUIRE(out);
    for (MTL::Buffer* b : {out.get(), ray_buffer.get(), count_buffer.get()}) {
        submission.make_resident(b);
    }
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(7);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);

    const metal::FrameSlot frame = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    if (moved != nullptr) {
        const std::span<contracts::Transform> placed = transforms.transforms(frame.slot);
        std::copy(moved->begin(), moved->end(), placed.begin());
        acceleration.update(encoder, frame.slot, placed);
    }
    table->setAddress(out->gpuAddress(), 0);
    table->setAddress(ray_buffer->gpuAddress(), 1);
    table->setAddress(count_buffer->gpuAddress(), 2);
    table->setResource(acceleration.resource(frame.slot), 3);
    table->setAddress(buffers.addresses().shapes, 4);
    table->setAddress(transforms.address(frame.slot), 5);
    table->setAddress(buffers.addresses().boxes, 6);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(frame.sequence);

    const auto* hits = static_cast<const ProbeHit*>(out->contents());
    return {hits, hits + count};
}

ProbeRay ray(float ox, float oy, float oz, float dx, float dy, float dz) {
    const float n = std::sqrt(dx * dx + dy * dy + dz * dz);
    return ProbeRay{{ox, oy, oz, 0.0f}, {dx / n, dy / n, dz / n, 0.0f}};
}

// A tiny firefly and a sphere wider than one, and a box: every scale the
// structure carries rays into object space at.
constexpr const char* scales = R"(
[camera]
position = [0, 0, 10]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glow]
kind = "emissive"
radiance = [1, 1, 1]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[[shapes]]
kind = "sphere"
center = [0, 0, -5]
radius = 0.03
material = "glow"
motion = { kind = "wander", reach = 0.01, speed = 0.01, seed = 1 }
[[shapes]]
kind = "sphere"
center = [20, 0, 0]
radius = 7.5
material = "matte"
[[shapes]]
kind = "box"
min = [-1, 9, -2]
max = [1, 11, 2]
material = "matte"
)";

}  // namespace

TEST_CASE("a hit, tested in a shape's own space, is at the world's distance, and names its shape") {
    const scene::SceneDescription scene = scene::parse(scales, "scales");
    metal::Device device;
    metal::Submission submission(device);
    metal::SceneBuffers buffers(device, submission, scene);
    // Still: the shapes where the file puts them.
    metal::ShapeTransforms transforms(device, submission, scene.shapes.transforms, false);
    metal::SceneAcceleration acceleration(device, submission, scene.shapes, {});

    const std::vector<ProbeRay> rays = {
        ray(0, 0, 0, 0, 0, -1),     // the firefly, radius 0.03, 5 away: t = 4.97
        ray(0, 0, 0, 1, 0, 0),      // the wide sphere, radius 7.5, 20 away: t = 12.5
        ray(0, 0, 0, 0, 1, 0),      // the box, its bottom face at y = 9
        ray(0, 0, 0, 0, -1, 0),     // nothing
        ray(0, 0.02f, 0, 0, 0, -1), // the firefly off its middle: sqrt(0.03^2 - 0.02^2) short of 5
    };
    const std::vector<ProbeHit> hits = probe(device, submission, buffers, transforms, acceleration, rays,
                                             nullptr);
    CHECK(hits[0].t == doctest::Approx(4.97).epsilon(1e-6));
    CHECK(hits[0].shape == 0.0f);
    CHECK(hits[1].t == doctest::Approx(12.5).epsilon(1e-6));
    CHECK(hits[1].shape == 1.0f);
    CHECK(hits[2].t == doctest::Approx(9.0).epsilon(1e-6));
    CHECK(hits[2].shape == 2.0f);
    CHECK(hits[3].t == -1.0f);
    CHECK(hits[4].t == doctest::Approx(5.0 - std::sqrt(0.03 * 0.03 - 0.02 * 0.02)).epsilon(1e-5));
}

TEST_CASE("a moved shape is hit where it moved to, and not where it was") {
    const scene::SceneDescription scene = scene::parse(scales, "scales");
    metal::Device device;
    metal::Submission submission(device);
    metal::SceneBuffers buffers(device, submission, scene);
    metal::ShapeTransforms transforms(device, submission, scene.shapes.transforms, true);
    const std::vector<std::uint32_t> moving = {0};
    metal::SceneAcceleration acceleration(device, submission, scene.shapes, moving);

    // Each slot in turn, so both structures are built and traced.
    for (int frame = 0; frame < 4; ++frame) {
        std::vector<contracts::Transform> moved = scene.shapes.transforms;
        const float x = 3.0f + static_cast<float>(frame);
        moved[0] = contracts::moved_to(moved[0], {x, 0.0f, -5.0f});
        const std::vector<ProbeRay> rays = {
            ray(x, 0, 0, 0, 0, -1),  // where it is now
            ray(0, 0, 0, 0, 0, -1),  // where it was at rest
        };
        const std::vector<ProbeHit> hits = probe(device, submission, buffers, transforms, acceleration, rays,
                                                 &moved);
        INFO("frame " << frame);
        CHECK(hits[0].t == doctest::Approx(4.97).epsilon(1e-6));
        CHECK(hits[0].shape == 0.0f);
        CHECK(hits[1].t == -1.0f);
    }
}

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

TEST_CASE("a moving scene's image holds one instant; a still scene's, any") {
    const frame::Extent size{32, 24};
    const auto frame_at = [](const scene::SceneDescription& s, std::uint64_t index, double time) {
        return frame::FrameInputs{.time = frame::Seconds(time), .index = index, .accumulated_since = 0,
                                  .camera = s.camera};
    };
    {
        const scene::SceneDescription moving = scene::parse(brass(), "moving");
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, graph("\"path\""), &moving);
        metal::Offscreen target(device, submission, size);
        (void)metal::render_to_offscreen(submission, target, renderer, frame_at(moving, 0, 1.5));
        // Another sample of the same instant joins it.
        (void)metal::render_to_offscreen(submission, target, renderer, frame_at(moving, 1, 1.5));
        // A frame of another instant is refused, not averaged in.
        CHECK_THROWS_WITH_AS(metal::render_to_offscreen(submission, target, renderer, frame_at(moving, 2, 1.6)),
                             doctest::Contains("an image holds one instant"), metal::Error);
        // Starting over at the new instant is fine.
        auto restart = frame_at(moving, 2, 1.6);
        restart.accumulated_since = 2;
        const std::uint64_t sequence = metal::render_to_offscreen(submission, target, renderer, restart);
        (void)submission.wait_until_complete(sequence);
        CHECK(renderer.non_finite_samples() == 0);
    }
    {
        const scene::SceneDescription still = scene::parse(brass(nullptr, 0.0), "still");
        scene::SceneDescription frozen = still;
        frozen.animation = {};  // the same scene, nothing moving
        metal::Device device;
        metal::Submission submission(device);
        metal::Renderer renderer(device, submission, graph("\"path\""), &frozen);
        metal::Offscreen target(device, submission, size);
        (void)metal::render_to_offscreen(submission, target, renderer, frame_at(frozen, 0, 1.5));
        const std::uint64_t sequence =
            metal::render_to_offscreen(submission, target, renderer, frame_at(frozen, 1, 9.0));
        (void)submission.wait_until_complete(sequence);
        CHECK(renderer.non_finite_samples() == 0);
    }
}
