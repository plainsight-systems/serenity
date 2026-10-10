// The camera's rays on the GPU (tests/gpu/kernels/camera_probe.metal),
// held to contracts/camera.h: through a lens, every ray through a point of
// the image, wherever on the lens it starts, meets the same point of the
// plane of focus, and starts on the lens; without one, the pinhole's ray,
// exactly.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/camera/thin_lens.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"
#include "support/references.h"
#include "support/vector.h"

using namespace serenity;
using tests::Binding;
using tests::CameraProbeRay;
using tests::CameraQuery;
using tests::Vec3;

namespace {

// The image the rays are made for.
constexpr frame::Extent image{1600, 900};

std::vector<CameraProbeRay> rays(const contracts::CameraData& camera, const std::vector<CameraQuery>& queries) {
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(queries.size());
    return gpu.run<CameraProbeRay>("camera_probe", queries.size(),
                                   {Binding::of(queries), Binding::of(count), Binding::of(camera),
                                    Binding::of(tests::CameraImage{image.width, image.height})});
}

double uniform(std::uint64_t i, std::uint64_t which) {
    return animation::draw(0xCA3u, i, which);
}

// A point of the image and of the lens, drawn from i.
CameraQuery query(std::uint64_t pixel, std::uint64_t lens) {
    return {static_cast<float>(image.width * uniform(pixel, 0)), static_cast<float>(image.height * uniform(pixel, 1)),
            static_cast<float>(uniform(lens, 2)), static_cast<float>(uniform(lens, 3))};
}

contracts::Camera framed_camera(float radius) {
    contracts::Camera c{{0.0f, 0.93f, 1.45f}, {0.0f, 0.87f, 0.0f}, {0.0f, 1.0f, 0.0f}, 38.0f};
    c.lens_radius = radius;
    c.focus_distance = 1.4f;
    return c;
}

}  // namespace

TEST_CASE("through a lens, every ray through an image point meets one point of the plane of focus") {
    constexpr float lens_radius = 0.012f;
    constexpr std::uint64_t points = 64;
    constexpr std::uint64_t per_point = 16;  // lens points through each image point
    const contracts::CameraData data = camera::shader_form(framed_camera(lens_radius), image);
    std::vector<CameraQuery> queries;
    queries.reserve(points * per_point);
    for (std::uint64_t p = 0; p < points; ++p) {
        for (std::uint64_t l = 0; l < per_point; ++l) {
            queries.push_back(query(p, p * per_point + l));
        }
    }
    const std::vector<CameraProbeRay> traced = rays(data, queries);
    const Vec3 origin = tests::vec(data.origin);
    const Vec3 forward = tests::vec(data.forward);
    double widest = 0.0;
    double farthest_start = 0.0;
    double farthest_off_lens = 0.0;
    for (std::size_t p = 0; p < points; ++p) {
        Vec3 first{};
        for (std::size_t l = 0; l < per_point; ++l) {
            const CameraProbeRay& r = traced[p * per_point + l];
            const Vec3 start = tests::vec(r.origin);
            const Vec3 direction = tests::vec(r.direction);
            // Where it meets the plane square to forward, focus_distance ahead.
            const double along = dot(direction, forward);
            REQUIRE(along > 0.0);
            const Vec3 offset = start - origin;
            const Vec3 hit = start + ((data.focus_distance - dot(offset, forward)) / along) * direction;
            if (l == 0) {
                first = hit;
            }
            widest = std::max(widest, tests::length(hit - first));
            // It starts on the lens: in the plane through the origin square
            // to forward, within the radius.
            farthest_off_lens = std::max(farthest_off_lens, std::abs(dot(offset, forward)));
            farthest_start = std::max(farthest_start, tests::length(offset));
        }
    }
    INFO("spread at the plane of focus " << widest << " m; farthest start " << farthest_start << " m");
    CHECK(widest < 1e-5);
    CHECK(farthest_off_lens < 1e-6);
    CHECK(farthest_start <= lens_radius * (1.0 + 1e-5));
    CHECK(farthest_start > 0.010);  // the lens is used, to its edge
}

TEST_CASE("without a lens, the ray is the pinhole's, whatever lens point is given") {
    const contracts::CameraData data = camera::shader_form(framed_camera(0.0f), image);
    std::vector<CameraQuery> queries;
    queries.reserve(256);
    for (std::uint64_t i = 0; i < 256; ++i) {
        queries.push_back(query(i, i));
    }
    const std::vector<CameraProbeRay> traced = rays(data, queries);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const Vec3 d = tests::camera_direction(data, queries[i].pixel_x, queries[i].pixel_y, image);
        CHECK(traced[i].origin.x == data.origin.x);
        CHECK(traced[i].origin.y == doctest::Approx(data.origin.y).scale(0).epsilon(1e-7));
        CHECK(traced[i].origin.z == data.origin.z);
        // A unit vector's components, each to 1e-5 of the vector's length.
        CHECK(traced[i].direction.x == doctest::Approx(d.x).scale(1.0).epsilon(1e-5));
        CHECK(traced[i].direction.y == doctest::Approx(d.y).scale(1.0).epsilon(1e-5));
        CHECK(traced[i].direction.z == doctest::Approx(d.z).scale(1.0).epsilon(1e-5));
    }
}
