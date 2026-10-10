// The camera's rays on the GPU (tests/gpu/kernels/camera_probe.metal),
// held to contracts/camera.h: through a lens, every ray through a point of
// the image, wherever on the lens it starts, meets the same point of the
// plane of focus, and starts on the lens; without one, the pinhole's ray,
// exactly.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/camera/thin_lens.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

namespace {

using Probe = std::array<float, 4>;

// Runs `kernel` over `points`, with `extra` bound at buffer 3 if given, and
// returns `stride` floats per point.
std::vector<float> run(const char* kernel, const std::vector<Probe>& points, std::size_t stride,
                       const void* extra = nullptr, std::size_t extra_bytes = 0) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline(kernel);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const auto count = static_cast<std::uint32_t>(points.size());
    const auto buffer = [&](const void* data, std::size_t bytes) {
        auto b = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(bytes, 16), MTL::ResourceStorageModeShared));
        REQUIRE(b);
        if (data != nullptr) {
            std::memcpy(b->contents(), data, bytes);
        }
        submission.make_resident(b.get());
        return b;
    };
    auto out = buffer(nullptr, points.size() * stride * sizeof(float));
    auto in = buffer(points.data(), points.size() * sizeof(Probe));
    auto n = buffer(&count, 4);
    auto more = buffer(extra, extra_bytes);
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(4);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    MTL::Buffer* bound[] = {out.get(), in.get(), n.get(), more.get()};
    for (std::size_t i = 0; i < 4; ++i) {
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
    std::vector<float> result(points.size() * stride);
    std::memcpy(result.data(), out->contents(), result.size() * sizeof(float));
    return result;
}

struct Ray {
    std::array<double, 3> origin, direction;
};

std::vector<Ray> rays(const contracts::CameraData& camera, const std::vector<Probe>& queries) {
    const std::vector<float> raw = run("camera_probe", queries, 8, &camera, sizeof(camera));
    std::vector<Ray> result(queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            result[i].origin[k] = raw[8 * i + k];
            result[i].direction[k] = raw[8 * i + 4 + k];
        }
    }
    return result;
}

double uniform(std::uint64_t i, std::uint64_t which) {
    return animation::draw(0xCA3u, i, which);
}

double dot(const std::array<double, 3>& a, contracts::Float3 b) {
    return a[0] * b.x + a[1] * b.y + a[2] * b.z;
}

contracts::Camera framed_camera(float radius) {
    contracts::Camera c{{0.0f, 0.93f, 1.45f}, {0.0f, 0.87f, 0.0f}, {0.0f, 1.0f, 0.0f}, 38.0f};
    c.lens_radius = radius;
    c.focus_distance = 1.4f;
    return c;
}

}  // namespace

TEST_CASE("through a lens, every ray through an image point meets one point of the plane of focus") {
    const contracts::CameraData data = camera::shader_form(framed_camera(0.012f), {1600, 900});
    std::vector<Probe> queries;
    for (std::uint64_t p = 0; p < 64; ++p) {
        const float x = float(1600.0 * uniform(p, 0)), y = float(900.0 * uniform(p, 1));
        for (std::uint64_t l = 0; l < 16; ++l) {
            queries.push_back({x, y, float(uniform(p * 16 + l, 2)), float(uniform(p * 16 + l, 3))});
        }
    }
    const std::vector<Ray> traced = rays(data, queries);
    double widest = 0.0, farthest_start = 0.0;
    for (std::size_t p = 0; p < 64; ++p) {
        std::array<double, 3> first{};
        for (std::size_t l = 0; l < 16; ++l) {
            const Ray& r = traced[p * 16 + l];
            // Where it meets the plane square to forward, focus_distance ahead.
            const double along = dot(r.direction, data.forward);
            REQUIRE(along > 0.0);
            std::array<double, 3> offset{r.origin[0] - data.origin.x, r.origin[1] - data.origin.y,
                                         r.origin[2] - data.origin.z};
            const double t = (data.focus_distance - dot(offset, data.forward)) / along;
            const std::array<double, 3> hit{r.origin[0] + t * r.direction[0], r.origin[1] + t * r.direction[1],
                                            r.origin[2] + t * r.direction[2]};
            if (l == 0) {
                first = hit;
            }
            widest = std::max(widest, std::hypot(hit[0] - first[0], hit[1] - first[1], hit[2] - first[2]));
            // It starts on the lens: in the plane through the origin square
            // to forward, within the radius.
            CHECK(std::abs(dot(offset, data.forward)) < 1e-6);
            farthest_start = std::max(farthest_start, std::hypot(offset[0], offset[1], offset[2]));
        }
    }
    INFO("spread at the plane of focus " << widest << " m; farthest start " << farthest_start << " m");
    CHECK(widest < 1e-5);
    CHECK(farthest_start <= 0.012 * (1.0 + 1e-5));
    CHECK(farthest_start > 0.010);  // the lens is used, to its edge
}

TEST_CASE("without a lens, the ray is the pinhole's, whatever lens point is given") {
    const contracts::CameraData data = camera::shader_form(framed_camera(0.0f), {1600, 900});
    std::vector<Probe> queries;
    for (std::uint64_t i = 0; i < 256; ++i) {
        queries.push_back({float(1600.0 * uniform(i, 0)), float(900.0 * uniform(i, 1)), float(uniform(i, 2)),
                           float(uniform(i, 3))});
    }
    const std::vector<Ray> traced = rays(data, queries);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const double sx = 2.0 * queries[i][0] / 1600.0 - 1.0, sy = 1.0 - 2.0 * queries[i][1] / 900.0;
        std::array<double, 3> d{data.forward.x + sx * data.right.x + sy * data.up.x,
                                data.forward.y + sx * data.right.y + sy * data.up.y,
                                data.forward.z + sx * data.right.z + sy * data.up.z};
        const double n = std::hypot(d[0], d[1], d[2]);
        CHECK(traced[i].origin[0] == data.origin.x);
        CHECK(traced[i].origin[1] == doctest::Approx(data.origin.y).scale(0).epsilon(1e-7));
        CHECK(traced[i].origin[2] == data.origin.z);
        for (int k = 0; k < 3; ++k) {
            // A unit vector's component, to 1e-5 of the vector's length.
            CHECK(traced[i].direction[k] == doctest::Approx(d[k] / n).scale(1.0).epsilon(1e-5));
        }
    }
}
