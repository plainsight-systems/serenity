// The acceleration structure (metal/acceleration/scene_acceleration.h), on
// the GPU, probed ray by ray through the tracing loop (kernels/
// trace_probe.metal): a hit, tested in a shape's own space, is at the
// world's distance and names the shape, at any scale; and a moved shape is
// hit where it moved to and not where it was, through both frame slots.
//
// Its own dispatch, not the probe runner's (support/probe_runner.h): it
// binds the structure as a resource, and records the structure's update in
// the same encoder as the trace, as a frame does.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "core/scene/scene.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"
#include "serenity/metallib/probes.h"
#include "support/vector.h"

using namespace serenity;
using tests::TraceHit;
using tests::TraceRay;

namespace {

// A scene on the GPU, as a frame traces it: its arrays, its shapes'
// transforms and its structure.
struct Traced {
    metal::Device& device;
    metal::Submission& submission;
    const metal::SceneBuffers& buffers;
    metal::ShapeTransforms& transforms;
    metal::SceneAcceleration& acceleration;
};

// Traces `rays` through the structure for the next frame's slot, in one
// command buffer; if `moved` is given, first writes them as the slot's
// transforms and records the structure's update, as a frame does.
std::vector<TraceHit> probe(const Traced& traced, const std::vector<TraceRay>& rays,
                            const std::optional<std::vector<contracts::Transform>>& moved) {
    metal::Library library(traced.device, metallib::probes);
    const auto pipeline = library.compute_pipeline("trace_probe");
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = traced.device.handle();
    const auto count = static_cast<std::uint32_t>(rays.size());
    auto out = NS::TransferPtr(mtl->newBuffer(count * sizeof(TraceHit), MTL::ResourceStorageModeShared));
    REQUIRE(out);
    auto ray_buffer =
        NS::TransferPtr(mtl->newBuffer(rays.data(), count * sizeof(TraceRay), MTL::ResourceStorageModeShared));
    REQUIRE(ray_buffer);
    auto count_buffer = NS::TransferPtr(mtl->newBuffer(&count, sizeof count, MTL::ResourceStorageModeShared));
    REQUIRE(count_buffer);
    for (MTL::Buffer* b : {out.get(), ray_buffer.get(), count_buffer.get()}) {
        traced.submission.make_resident(b);
    }
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(7);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    INFO("argument table: " << tests::reason(error));
    REQUIRE(table);

    const metal::FrameSlot slot = traced.submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    if (moved) {
        const std::span<contracts::Transform> placed = traced.transforms.transforms(slot.slot);
        std::ranges::copy(*moved, placed.begin());
        traced.acceleration.update(encoder, slot.slot, placed);
    }
    // The bindings kernels/trace_probe.metal lists.
    table->setAddress(out->gpuAddress(), 0);
    table->setAddress(ray_buffer->gpuAddress(), 1);
    table->setAddress(count_buffer->gpuAddress(), 2);
    table->setResource(traced.acceleration.resource(slot.slot), 3);
    table->setAddress(traced.buffers.block().shapes, 4);
    table->setAddress(traced.transforms.address(slot.slot), 5);
    table->setAddress(traced.buffers.block().boxes, 6);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    traced.submission.commit();
    (void)traced.submission.wait_until_complete(slot.sequence);

    std::vector<TraceHit> hits(count);
    std::memcpy(hits.data(), out->contents(), count * sizeof(TraceHit));
    return hits;
}

// A ray from `origin` along the unit vector toward `toward`.
TraceRay ray(tests::Vec3 origin, tests::Vec3 toward) {
    const tests::Vec3 d = tests::normalized(toward);
    return TraceRay{{static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z)},
                    {static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z)}};
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

// The firefly, its radius and its center's distance from the origin.
constexpr double firefly_radius = 0.03;
constexpr double firefly_distance = 5.0;

}  // namespace

TEST_CASE("a hit, tested in a shape's own space, is at the world's distance, and names its shape") {
    const scene::SceneDescription description = scene::parse(scales, "scales");
    metal::Device device;
    metal::Submission submission(device);
    const metal::SceneBuffers buffers(device, submission, description);
    // Still: the shapes where the file puts them.
    metal::ShapeTransforms transforms(device, submission, description.shapes.transforms,
                                      metal::FrameArray::Copies::one);
    metal::SceneAcceleration acceleration(device, submission, description.shapes, {});

    const std::vector<TraceRay> rays = {
        ray({0, 0, 0}, {0, 0, -1}),      // the firefly, 5 away: t = 4.97
        ray({0, 0, 0}, {1, 0, 0}),       // the wide sphere, radius 7.5, 20 away: t = 12.5
        ray({0, 0, 0}, {0, 1, 0}),       // the box, its bottom face at y = 9
        ray({0, 0, 0}, {0, -1, 0}),      // nothing
        ray({0, 0.02, 0}, {0, 0, -1}),   // the firefly off its middle: sqrt(0.03^2 - 0.02^2) short of 5
    };
    const std::vector<TraceHit> hits =
        probe({device, submission, buffers, transforms, acceleration}, rays, std::nullopt);
    CHECK(hits[0].found == 1u);
    CHECK(hits[0].t == doctest::Approx(firefly_distance - firefly_radius).scale(0).epsilon(1e-6));
    CHECK(hits[0].shape == 0u);
    CHECK(hits[1].found == 1u);
    CHECK(hits[1].t == doctest::Approx(12.5).scale(0).epsilon(1e-6));
    CHECK(hits[1].shape == 1u);
    CHECK(hits[2].found == 1u);
    CHECK(hits[2].t == doctest::Approx(9.0).scale(0).epsilon(1e-6));
    CHECK(hits[2].shape == 2u);
    CHECK(hits[3].found == 0u);
    CHECK(hits[4].found == 1u);
    CHECK(hits[4].t ==
          doctest::Approx(firefly_distance - std::sqrt(firefly_radius * firefly_radius - 0.02 * 0.02))
              .scale(0)
              .epsilon(1e-5));
}

TEST_CASE("a moved shape is hit where it moved to, and not where it was") {
    const scene::SceneDescription description = scene::parse(scales, "scales");
    metal::Device device;
    metal::Submission submission(device);
    const metal::SceneBuffers buffers(device, submission, description);
    metal::ShapeTransforms transforms(device, submission, description.shapes.transforms,
                                      metal::FrameArray::Copies::per_frame);
    const std::vector<std::uint32_t> moving = {0};
    metal::SceneAcceleration acceleration(device, submission, description.shapes, moving);

    // Each slot in turn, so both structures are built and traced.
    for (int round = 0; round < 4; ++round) {
        std::vector<contracts::Transform> moved = description.shapes.transforms;
        const float x = 3.0f + static_cast<float>(round);
        moved[0] = contracts::moved_to(moved[0], {x, 0.0f, -5.0f});
        const std::vector<TraceRay> rays = {
            ray({x, 0, 0}, {0, 0, -1}),  // where it is now
            ray({0, 0, 0}, {0, 0, -1}),  // where it was at rest
        };
        const std::vector<TraceHit> hits = probe({device, submission, buffers, transforms, acceleration}, rays, moved);
        INFO("round " << round);
        CHECK(hits[0].found == 1u);
        CHECK(hits[0].t == doctest::Approx(firefly_distance - firefly_radius).scale(0).epsilon(1e-6));
        CHECK(hits[0].shape == 0u);
        CHECK(hits[1].found == 0u);
    }
}
