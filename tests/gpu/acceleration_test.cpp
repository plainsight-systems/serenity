// The acceleration structure (metal/acceleration/scene_acceleration.h), on
// the GPU, probed ray by ray through the tracing loop (kernels/
// trace_probe.metal): a hit, tested in a shape's own space, is at the
// world's distance and names the shape, at any scale; and a moved shape is
// hit where it moved to and not where it was, through both frame slots.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "core/scene/scene.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
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
std::vector<ProbeHit> probe(metal::Device& device, metal::Submission& submission, const metal::SceneBuffers& buffers, metal::ShapeTransforms& transforms,
                            metal::SceneAcceleration& acceleration, const std::vector<ProbeRay>& rays,
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
    table->setAddress(buffers.block().shapes, 4);
    table->setAddress(transforms.address(frame.slot), 5);
    table->setAddress(buffers.block().boxes, 6);
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
