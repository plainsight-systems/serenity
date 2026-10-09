// The sphere light's cone (metal/lights/sphere_light.metal.h), run on the
// GPU: shadow rays must cover all of the light as seen from a point, out to
// its rim, and nothing beyond it. A near light is the case that matters: at
// a distance of twice its radius its cone is 30 degrees, where a disk
// through its center would cover only 26.6.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>

#include <doctest/doctest.h>

#include "core/lights/sphere_light.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

TEST_CASE("shadow rays toward a near sphere light span its whole cone, and end on its surface") {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline("light_cone");

    constexpr std::uint32_t n = 64;
    const lights::SphereLightData light{{0.0f, 2.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}, 0};
    const float point_and_n[4] = {0.0f, 0.0f, 0.0f, static_cast<float>(n)};  // d = 2r

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    auto out = NS::TransferPtr(mtl->newBuffer(n * n * 16, MTL::ResourceStorageModeShared));
    auto light_buffer = NS::TransferPtr(mtl->newBuffer(sizeof(light), MTL::ResourceStorageModeShared));
    auto point_buffer = NS::TransferPtr(mtl->newBuffer(sizeof(point_and_n), MTL::ResourceStorageModeShared));
    REQUIRE(out);
    REQUIRE(light_buffer);
    REQUIRE(point_buffer);
    std::memcpy(light_buffer->contents(), &light, sizeof(light));
    std::memcpy(point_buffer->contents(), point_and_n, sizeof(point_and_n));
    for (MTL::Buffer* buffer : {out.get(), light_buffer.get(), point_buffer.get()}) {
        submission.make_resident(buffer);
    }

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(3);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    table->setAddress(out->gpuAddress(), 0);
    table->setAddress(light_buffer->gpuAddress(), 1);
    table->setAddress(point_buffer->gpuAddress(), 2);

    const auto frame = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(n * n, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(frame.sequence);

    // The cone's half-angle: asin(r / d) = 30 degrees.
    const double cone = std::asin(0.5);
    double widest = 0.0;
    double worst_surface_error = 0.0;
    const auto* samples = static_cast<const float*>(out->contents());
    for (std::uint32_t i = 0; i < n * n; ++i) {
        const double x = samples[4 * i], y = samples[4 * i + 1], z = samples[4 * i + 2], t = samples[4 * i + 3];
        const double angle = std::acos(std::clamp(y / std::sqrt(x * x + y * y + z * z), -1.0, 1.0));
        widest = std::max(widest, angle);
        // The point t along the ray lies on the light's surface.
        const double px = t * x, py = t * y - 2.0, pz = t * z;
        worst_surface_error = std::max(worst_surface_error, std::abs(std::sqrt(px * px + py * py + pz * pz) - 1.0));
    }
    INFO("widest " << widest * 180.0 / std::numbers::pi << " degrees, cone " << cone * 180.0 / std::numbers::pi);
    CHECK(widest <= cone + 1e-3);
    // The grid's outermost row of samples is half a cell from the rim, in
    // cos t: within a degree of it, and far past the 26.6 a center disk gives.
    CHECK(widest >= cone - 1.0 * std::numbers::pi / 180.0);
    CHECK(worst_surface_error < 1e-3);
}
