// The sphere light (metal/lights/sphere_light.metal.h), run on the GPU
// (tests/gpu/kernels/light_probe.metal): shadow rays must cover all of the
// light as seen from a point, out to its rim, and nothing beyond it, drawn
// uniformly over its cone as the pdf says, and a small light far off must
// keep its solid angle. A near light is the case that matters: at a
// distance of twice its radius its cone is 30 degrees, where a disk through
// its center would cover only 26.6.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/transform.h"
#include "core/lights/sphere_light.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"
#include "support/vector.h"

using namespace serenity;
using tests::Binding;
using tests::LightQuery;
using tests::Vec3;

namespace {

constexpr lights::SphereLightData white_light{.radiance = {1.0f, 1.0f, 1.0f}, .shape = 0};

// A light of radius 1 at (0, 2, 0), seen from the origin: d = 2r, its cone's
// half-angle asin(1 / 2), 30 degrees.
constexpr contracts::Float3 near_center{0.0f, 2.0f, 0.0f};
constexpr float near_radius = 1.0f;
const double near_cone = std::asin(near_radius / near_center.y);

}  // namespace

TEST_CASE("shadow rays toward a near sphere light span its whole cone, and end on its surface") {
    constexpr std::uint32_t n = 64;
    const contracts::Transform placed = contracts::placed(near_center, near_radius);
    tests::ProbeRunner gpu;
    const std::vector<tests::LightDraw> drawn =
        gpu.run<tests::LightDraw>("light_cone", n * n,
                                  {Binding::of(white_light), Binding::of(LightQuery{{0.0f, 0.0f, 0.0f}, n}),
                                   Binding::of(placed)});

    double widest = 0.0;
    double worst_surface_error = 0.0;
    const Vec3 center = tests::vec(near_center);
    for (const tests::LightDraw& d : drawn) {
        const Vec3 direction = tests::vec(d.direction);
        widest = std::max(widest, std::acos(std::clamp(direction.y / tests::length(direction), -1.0, 1.0)));
        // The point `distance` along the ray lies on the light's surface.
        const Vec3 on_light = static_cast<double>(d.distance) * direction;
        worst_surface_error = std::max(worst_surface_error, std::abs(tests::length(on_light - center) - near_radius));
    }
    INFO("widest " << widest * 180.0 / std::numbers::pi << " degrees, cone " << near_cone * 180.0 / std::numbers::pi);
    CHECK(widest <= near_cone + 1e-3);
    // The grid's outermost row of samples is half a cell from the rim, in
    // cos t: within a degree of it, and far past the 26.6 a center disk gives.
    CHECK(widest >= near_cone - 1.0 * std::numbers::pi / 180.0);
    CHECK(worst_surface_error < 1e-3);
}

TEST_CASE("a near light's samples are uniform over its cone, as its pdf says, and each pdf() agrees") {
    // A million draws of the emitter's sample() from the origin, as a path
    // draws them, binned by cos t, uniform on [cos a, 1], and by azimuth
    // about the direction to the light's center, uniform on [0, 2 pi):
    // 16 x 16 bins of equal solid angle, each expecting a 256th. Pearson's
    // chi-square over them, as for the BSDFs (bsdf_test.cpp). A sampler
    // that bunched its draws toward the middle or the rim, or to one side,
    // would pass the cone's bounds above and fail here.
    constexpr std::uint32_t count = 1u << 20;
    constexpr std::uint32_t side = 16;
    const contracts::Transform placed = contracts::placed(near_center, near_radius);
    tests::ProbeRunner gpu;
    const std::vector<tests::LightSampleDraw> drawn =
        gpu.run<tests::LightSampleDraw>("light_samples", count,
                                        {Binding::of(white_light), Binding::of(LightQuery{{0.0f, 0.0f, 0.0f}, count}),
                                         Binding::of(placed)});

    const double cos_max = std::cos(near_cone);
    const double solid_angle = 2.0 * std::numbers::pi * (1.0 - cos_max);
    std::vector<double> counted(std::size_t{side} * side, 0.0);
    std::uint32_t wrong_pdf = 0;
    std::uint32_t outside = 0;
    for (const tests::LightSampleDraw& d : drawn) {
        // About the axis to the center, +y: cos t is y, the azimuth in x-z.
        const Vec3 direction = tests::vec(d.direction);
        const double cos_t = direction.y;
        const double phi = std::atan2(direction.z, direction.x) + std::numbers::pi;
        const bool in_cone = cos_t >= cos_max - 1e-6 && cos_t <= 1.0 + 1e-6;
        outside += in_cone ? 0u : 1u;
        wrong_pdf += std::abs(d.pdf * solid_angle - 1.0) < 1e-4 && std::abs(d.pdf_asked * solid_angle - 1.0) < 1e-4
                         ? 0u
                         : 1u;
        const auto row = std::min(side - 1, static_cast<std::uint32_t>((1.0 - cos_t) / (1.0 - cos_max) * side));
        const auto column = std::min(side - 1, static_cast<std::uint32_t>(phi / (2.0 * std::numbers::pi) * side));
        counted[std::size_t{row} * side + column] += 1.0;
    }
    CHECK(outside == 0);
    CHECK(wrong_pdf == 0);

    const double expected = static_cast<double>(count) / (side * side);
    double chi2 = 0.0;
    for (const double c : counted) {
        chi2 += (c - expected) * (c - expected) / expected;
    }
    const double dof = side * side - 1.0;
    INFO("chi^2 / dof = " << chi2 / dof << " over " << side * side << " bins");
    CHECK(chi2 / dof < 1.0 + 5.0 * std::sqrt(2.0 / dof));
}

TEST_CASE("a small light far off keeps its solid angle: no cancellation to zero") {
    // r / d = 1e-4: 1 - sqrt(1 - 1e-8) is 0 in float, which would make the
    // pdf infinite and the light vanish; sin^2 / (1 + cos) keeps it.
    const contracts::Transform placed = contracts::placed({0.0f, 1000.0f, 0.0f}, 0.1f);
    tests::ProbeRunner gpu;
    const tests::LightFar far =
        gpu.run<tests::LightFar>("light_far", 1,
                                 {Binding::of(white_light), Binding::of(LightQuery{{0.0f, 0.0f, 0.0f}, 1u}),
                                  Binding::of(placed)})
            .at(0);
    const double sin2 = 1e-8;
    const double solid_angle = 2.0 * std::numbers::pi * sin2 / (1.0 + std::sqrt(1.0 - sin2));
    INFO("pdf " << far.sample_pdf << ", expected " << 1.0 / solid_angle);
    CHECK(std::isfinite(far.sample_pdf));
    CHECK(far.sample_pdf == doctest::Approx(1.0 / solid_angle).scale(0).epsilon(1e-4));
    CHECK(far.middle_pdf == doctest::Approx(1.0 / solid_angle).scale(0).epsilon(1e-4));  // its middle, inside
    CHECK(far.aside_pdf == 0.0f);                                                          // well aside, outside
    CHECK(far.sample_distance == doctest::Approx(999.9).scale(0).epsilon(1e-5));           // to its surface
}
