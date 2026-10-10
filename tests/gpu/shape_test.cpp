// The shapes' surface interaction on the GPU (tests/gpu/kernels/
// shape_probe.metal), held to contract 1: the point in the shape's own
// coordinates is where the hit is on its geometry before its transform, and
// the interior is the shape's medium (contract 12).

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

namespace {

using Query = std::array<float, 4>;

// interaction_probe over `queries`, the scene's shape arrays bound beside.
std::vector<contracts::SurfaceInteraction> interactions(const scene::SceneDescription& scene,
                                                        const std::vector<Query>& queries) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline("interaction_probe");
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const auto count = static_cast<std::uint32_t>(queries.size());
    const auto buffer = [&](const void* data, std::size_t bytes) {
        auto b = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(bytes, 16), MTL::ResourceStorageModeShared));
        REQUIRE(b);
        if (data != nullptr) {
            std::memcpy(b->contents(), data, bytes);
        }
        submission.make_resident(b.get());
        return b;
    };
    const auto& shapes = scene.shapes;
    NS::SharedPtr<MTL::Buffer> bound[] = {
        buffer(nullptr, queries.size() * sizeof(contracts::SurfaceInteraction)),
        buffer(queries.data(), queries.size() * sizeof(Query)),
        buffer(&count, 4),
        buffer(shapes.records.data(), shapes.records.size() * sizeof(shapes::ShapeRecord)),
        buffer(shapes.transforms.data(), shapes.transforms.size() * sizeof(contracts::Transform)),
        buffer(shapes.boxes.data(), shapes.boxes.size() * sizeof(shapes::BoxData)),
    };
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(6);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    for (std::size_t i = 0; i < 6; ++i) {
        table->setAddress(bound[i]->gpuAddress(), i);
    }
    const auto frame = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(frame.sequence);
    std::vector<contracts::SurfaceInteraction> out(queries.size());
    std::memcpy(out.data(), bound[0]->contents(), out.size() * sizeof(contracts::SurfaceInteraction));
    return out;
}

}  // namespace

TEST_CASE("a surface interaction carries the point on its shape's own geometry, and the medium inside") {
    const scene::SceneDescription scene = scene::parse(R"(
[camera]
position = [0, 0, 5]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glass]
kind = "dielectric"
ior = 1.5
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[media.tint]
kind = "absorbing"
tint = [0.5, 0.5, 0.5]
tint_distance = 1
[[shapes]]
kind = "sphere"
center = [1, 2, 3]
radius = 0.5
material = "glass"
interior = "tint"
[[shapes]]
kind = "box"
min = [-2, -1, -1]
max = [-1, 0, 0]
material = "matte"
)",
                                                       "s");
    // On the sphere, 0.5 from (1, 2, 3) along (0.6, 0, 0.8): on the unit
    // sphere at (0.6, 0, 0.8). On the box's top face, its own corners being
    // the world's.
    const std::vector<Query> queries = {{1.3f, 2.0f, 3.4f, 0.0f}, {-1.5f, 0.0f, -0.5f, 1.0f}};
    const auto out = interactions(scene, queries);
    CHECK(out[0].object_position.x == doctest::Approx(0.6f).scale(0).epsilon(1e-5));
    CHECK(out[0].object_position.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(out[0].object_position.z == doctest::Approx(0.8f).scale(0).epsilon(1e-5));
    CHECK(out[0].interior == 0u);
    CHECK(out[0].geometric_normal.x == doctest::Approx(0.6f).scale(0).epsilon(1e-5));
    CHECK(out[1].object_position.x == doctest::Approx(-1.5f).scale(0).epsilon(1e-6));
    CHECK(out[1].object_position.y == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(out[1].object_position.z == doctest::Approx(-0.5f).scale(0).epsilon(1e-6));
    CHECK(out[1].interior == contracts::no_medium);
}
