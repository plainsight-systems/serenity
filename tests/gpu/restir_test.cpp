// ReSTIR DI without reuse (metal/passes/restir_di/restir_di.metal.h, steps
// 1 to 3 and 6), on the GPU, each piece against what it must be:
//
//   - the equal-area octahedral map (metal/math/warp.metal.h): its round
//     trip within the contract's stated angle over the whole sphere, and
//     equal area, uniform draws over the sphere landing uniformly on the
//     square;
//   - the packed BSDF (metal/materials/bsdf.metal.h): every kind back
//     within contract 2's stated tolerances, and the fields a kind does not
//     read 0;
//   - the light point (metal/lights/emitter.metal.h): drawn and evaluated
//     at the same point it is the draw's direction, distance, radiance and
//     density, and its density in area measure is the closed form's;
//   - the target (metal/integrator/target.metal.h): closed forms for a
//     Lambert floor under one sphere light;
//   - the reservoir (metal/reuse/reservoir.metal.h): fed exp(score), it
//     keeps key i with probability softmax(score)_i, by Pearson's
//     chi-square, in any order of the stream, and the test refuses
//     exp(2 score); finished, W = w_sum / (M p̂);
//   - the shade's rule for when a reservoir's own visibility test stands;
//   - the renderer: resampled importance sampling's floor converges to the
//     closed form the path tracer's does, with M = 1 (next event estimation)
//     and M = 8 alike, M = 8 with less noise; visibility reuse leaves a
//     hidden light nothing; the same inputs give the same image; and a
//     graph with the merges is refused by name.
//
// Every statistical check is deterministic: its numbers are the sampler's,
// keyed by the stream's index, so it passes or fails the same every run.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/transform.h"
#include "core/frame/graph_file.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "core/passes/restir_di.h"
#include "core/scene/scene.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_images.h"
#include "metal/frame/frame_resources.h"
#include "metal/frame/renderer.h"
#include "metal/passes/restir_initial/restir_initial.h"
#include "metal/passes/restir_shade/restir_shade.h"
#include "serenity/metallib/shaders.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/text.h"
#include "support/vector.h"

using namespace serenity;
using tests::Binding;
using tests::Vec3;

namespace {

// ---- statistics ----------------------------------------------------------

// Q(a, x), the regularized upper incomplete gamma function (Numerical
// Recipes' gammq: the series below a + 1, the continued fraction above).
double upper_gamma(double a, double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    const double log_front = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n < 1000 && std::abs(term) > std::abs(sum) * 1e-15; ++n) {
            term *= x / (a + n);
            sum += term;
        }
        return 1.0 - sum * std::exp(log_front);
    }
    constexpr double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        d = std::abs(d) < tiny ? tiny : d;
        c = b + an / c;
        c = std::abs(c) < tiny ? tiny : c;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::abs(step - 1.0) < 1e-15) {
            break;
        }
    }
    return std::exp(log_front) * h;
}

// Pearson's chi-square of `counted` against `expected`, and the chance of
// one at least as large under the expected distribution.
struct ChiSquare {
    double statistic = 0.0;
    std::size_t dof = 0;
    double p = 0.0;
};

ChiSquare chi_square(const std::vector<double>& counted, const std::vector<double>& expected) {
    REQUIRE(counted.size() == expected.size());
    ChiSquare result;
    for (std::size_t i = 0; i < counted.size(); ++i) {
        result.statistic += (counted[i] - expected[i]) * (counted[i] - expected[i]) / expected[i];
    }
    result.dof = counted.size() - 1;
    result.p = upper_gamma(0.5 * static_cast<double>(result.dof), 0.5 * result.statistic);
    return result;
}

// The significance every statistical check here is made at: a correct
// sampler fails one in a thousand seeds, and the seeds are fixed.
constexpr double significance = 1e-3;

// ---- the probes' one light -----------------------------------------------

// Sphere light record 0, of `radius` at `center`, `radiance` every channel,
// as the probes bind it (kernels/restir_probe.metal): the records, the
// spheres, the transforms, the glows, which light each shape is.
struct ProbeLight {
    contracts::Float3 center;
    float radius;
    float radiance;

    std::array<Binding, 5> bindings() const {
        return {Binding::of(std::vector<lights::LightRecord>{{lights::LightKind::sphere, 0u}}),
                Binding::of(std::vector<lights::SphereLightData>{{{radiance, radiance, radiance}, 0u}}),
                Binding::of(std::vector<contracts::Transform>{contracts::placed(center, radius)}),
                Binding::of(std::vector<float>{1.0f}), Binding::of(std::vector<std::uint32_t>{0u})};
    }
};

// The light's bindings, then `rest`.
std::vector<Binding> with_light(const ProbeLight& light, std::initializer_list<Binding> rest) {
    const std::array<Binding, 5> first = light.bindings();
    std::vector<Binding> all(first.begin(), first.end());
    all.insert(all.end(), rest.begin(), rest.end());
    return all;
}

// Runs `kernel` over `threads` with the light and `rest` bound from 1, its
// output of Out at 0.
template <typename Out>
std::vector<Out> run_with_light(tests::ProbeRunner& gpu, const char* kernel, std::size_t threads,
                                const ProbeLight& light, std::initializer_list<Binding> rest) {
    std::vector<Binding> bindings{Binding::zeroed(threads * sizeof(Out))};
    const std::vector<Binding> inputs = with_light(light, rest);
    bindings.insert(bindings.end(), inputs.begin(), inputs.end());
    return gpu.dispatch(kernel, threads, std::span<const Binding>(bindings)).template read<Out>(0);
}

double angle_between(Vec3 a, Vec3 b) {
    return std::acos(std::clamp(dot(tests::normalized(a), tests::normalized(b)), -1.0, 1.0));
}

// The angle between `a` and `b`, robust for tiny angles: atan2 of the cross
// product's length and the dot product.
double small_angle(Vec3 a, Vec3 b) {
    const Vec3 c{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    return std::atan2(tests::length(c), dot(a, b));
}

}  // namespace

// ---- the equal-area octahedral map -----------------------------------------

TEST_CASE("the equal-area octahedral map round-trips every direction within the contract's angle") {
    // A Fibonacci spiral of directions over the whole sphere, and the
    // places the map treats apart: the poles, the equator, the octants'
    // seams, and directions a hair off each.
    std::vector<contracts::Float3> directions;
    constexpr int spiral = 1 << 18;
    const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < spiral; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / spiral;
        const double r = std::sqrt(1.0 - z * z);
        directions.push_back({static_cast<float>(r * std::cos(golden * i)), static_cast<float>(r * std::sin(golden * i)),
                              static_cast<float>(z)});
    }
    for (const double hair : {0.0, 1e-7, 1e-5, 1e-3}) {
        for (const int sx : {-1, 1}) {
            for (const int sy : {-1, 1}) {
                for (const int sz : {-1, 1}) {
                    for (const Vec3 v : {Vec3{hair, hair, 1.0}, Vec3{1.0, hair, hair}, Vec3{hair, 1.0, hair},
                                         Vec3{1.0, 1.0, hair}, Vec3{1.0, hair, 1.0}, Vec3{1.0, 1.0, 1.0}}) {
                        const Vec3 u = tests::normalized(Vec3{sx * v.x, sy * v.y, sz * v.z});
                        directions.push_back({static_cast<float>(u.x), static_cast<float>(u.y), static_cast<float>(u.z)});
                    }
                }
            }
        }
    }
    const auto count = static_cast<std::uint32_t>(directions.size());
    tests::ProbeRunner gpu;
    const std::vector<tests::UnitTrip> trips =
        gpu.run<tests::UnitTrip>("unit_trip", count, {Binding::of(directions), Binding::of(count)});
    double worst = 0.0;
    double worst_length = 0.0;
    for (std::size_t i = 0; i < trips.size(); ++i) {
        worst = std::max(worst, small_angle(tests::vec(directions[i]), tests::vec(trips[i].decoded)));
        worst_length = std::max(worst_length, std::abs(tests::length(tests::vec(trips[i].decoded)) - 1.0));
    }
    MESSAGE("worst round trip " << worst << " radians over " << count << " directions; the contract's "
                             << contracts::unit_round_trip_error);
    CHECK(worst <= contracts::unit_round_trip_error);
    // The bound is the map's, not slack: a map twice as coarse, or a
    // decoding half a cell off, would pass it by no more than this.
    CHECK(worst >= 0.25 * contracts::unit_round_trip_error);
    CHECK(worst_length < 1e-6);
}

TEST_CASE("the equal-area octahedral map is equal-area: uniform draws fill the square's cells evenly") {
    // 2^22 draws uniform over the sphere, their codes binned 16 x 16 by
    // each coordinate's top four bits: every bin is the same area of the
    // sphere, so each expects 1 / 256 of the draws (the edge bins half a
    // code short, 1e-4 of a bin, far below the noise). And the test can
    // fail: draws by the cosine about +z, which are not uniform, are
    // refused at any significance.
    constexpr std::uint32_t draws = 1u << 22;
    constexpr std::size_t side = 16;
    tests::ProbeRunner gpu;
    const auto binned = [&](std::uint32_t distribution) {
        const std::vector<std::uint32_t> codes =
            gpu.run<std::uint32_t>("unit_codes", draws, {Binding::of(tests::UnitDraws{draws, distribution})});
        std::vector<double> counted(side * side, 0.0);
        for (const std::uint32_t code : codes) {
            counted[((code >> 28u) & 0xfu) * side + ((code >> 12u) & 0xfu)] += 1.0;
        }
        return chi_square(counted, std::vector<double>(side * side, double{draws} / (side * side)));
    };
    const ChiSquare uniform = binned(tests::draw_uniform);
    MESSAGE("uniform: chi^2 " << uniform.statistic << ", " << uniform.dof << " dof, p " << uniform.p);
    CHECK(uniform.p > significance);
    const ChiSquare cosine = binned(tests::draw_cosine);
    MESSAGE("cosine: chi^2 " << cosine.statistic << ", " << cosine.dof << " dof, p " << cosine.p);
    CHECK(cosine.p < 1e-12);
}

// ---- the packed BSDF -------------------------------------------------------

namespace {

// Contract 2's tolerance for a half: 2^-11 of the value, relatively, from
// half's least normal up; 2^-24, absolutely, below it.
bool within_half(double packed, double value) {
    const double tolerance = std::abs(value) >= double{contracts::packed_least_normal}
                                 ? double{contracts::packed_relative} * std::abs(value)
                                 : double{contracts::packed_absolute};
    return std::abs(packed - value) <= tolerance;
}

}  // namespace

TEST_CASE("a packed Bsdf unpacks within contract 2's tolerances, for every kind") {
    std::vector<contracts::Bsdf> bsdfs;
    // Normals over the sphere, colors and scalars over the ranges materials
    // resolve to, for each kind.
    for (std::uint32_t i = 0; i < 4096; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / 4096.0;
        const double phi = 2.399963 * i;
        const double r = std::sqrt(1.0 - z * z);
        contracts::Bsdf b{};
        b.normal = {static_cast<float>(r * std::cos(phi)), static_cast<float>(r * std::sin(phi)), static_cast<float>(z)};
        const float t = static_cast<float>(i % 97) / 96.0f;
        b.color = {0.02f + 0.97f * t, 0.5f * t + 0.01f, 1.0f - 0.9f * t};
        switch (i % 4) {
        case 0:
            b.kind = contracts::BsdfKind::lambert;
            break;
        case 1:
            b.kind = contracts::BsdfKind::conductor;
            b.alpha = 0.0001f + 0.9f * t * t;
            break;
        case 2:
            b.kind = contracts::BsdfKind::dielectric;
            b.color = {1.0f, 1.0f, 1.0f};
            b.ior = 1.3f + 0.5f * t;
            break;
        default:
            b.kind = contracts::BsdfKind::coated;
            b.ior = 1.4f + 0.3f * t;
            b.escape = 0.9f + 0.09f * t;
            break;
        }
        bsdfs.push_back(b);
    }
    // The contract's endpoints (contracts/bsdf.h), each held to its
    // tolerance in the loop below with the rest: the largest, 65504, in the
    // ior and a color, and its negative; half's least normal, 2^-14, where
    // the relative tolerance starts; and below it, where the absolute one
    // holds: values that round to subnormals, 1.5 x 2^-24, and positive
    // ones under 2^-25, which round to 0 and are raised to 2^-24 (1e-9, 60
    // times smaller, among them).
    const std::size_t endpoints = bsdfs.size();
    contracts::Bsdf largest{};
    largest.normal = {0.0f, 0.0f, 1.0f};
    largest.kind = contracts::BsdfKind::dielectric;
    largest.color = {contracts::packed_largest, -contracts::packed_largest, contracts::packed_least_normal};
    largest.ior = contracts::packed_largest;
    bsdfs.push_back(largest);
    contracts::Bsdf glassy_metal{};
    glassy_metal.normal = {0.0f, 1.0f, 0.0f};
    glassy_metal.kind = contracts::BsdfKind::conductor;
    glassy_metal.color = {1e-9f, 1.5f * contracts::packed_absolute, 0.4f * contracts::packed_absolute};
    glassy_metal.alpha = 1e-9f;
    bsdfs.push_back(glassy_metal);
    const auto count = static_cast<std::uint32_t>(bsdfs.size());

    tests::ProbeRunner gpu;
    const std::vector<contracts::Bsdf> back =
        gpu.run<contracts::Bsdf>("bsdf_round_trip", count, {Binding::of(bsdfs), Binding::of(count)});
    double worst_normal = 0.0;
    std::size_t outside = 0;
    for (std::size_t i = 0; i < bsdfs.size(); ++i) {
        const contracts::Bsdf& a = bsdfs[i];
        const contracts::Bsdf& b = back[i];
        CHECK(a.kind == b.kind);
        worst_normal = std::max(worst_normal, small_angle(tests::vec(a.normal), tests::vec(b.normal)));
        const bool colors = within_half(b.color.x, a.color.x) && within_half(b.color.y, a.color.y) &&
                            within_half(b.color.z, a.color.z);
        const bool scalars = within_half(b.alpha, a.alpha) && within_half(b.ior, a.ior) &&
                             within_half(b.escape, a.escape);
        outside += colors && scalars ? 0u : 1u;
        // What a kind does not read is 0, padding included.
        CHECK(b.padding[0] == 0u);
        CHECK(b.padding[1] == 0u);
    }
    MESSAGE("worst normal " << worst_normal << " radians");
    CHECK(worst_normal <= contracts::unit_round_trip_error);
    CHECK(outside == 0u);
    // The tolerance check can fail, in either regime: a value off by two
    // halves' steps, and one off by twice the absolute bound.
    CHECK_FALSE(within_half(0.75 * (1.0 + 4.0 * double{contracts::packed_relative}), 0.75));
    CHECK_FALSE(within_half(3.0 * double{contracts::packed_absolute}, 1e-9));
    // The endpoints exactly: 65504 and 2^-14 are halves; below 2^-25, 2^-24,
    // never 0.
    const contracts::Bsdf& at_largest = back[endpoints];
    CHECK(at_largest.ior == contracts::packed_largest);
    CHECK(at_largest.color.x == contracts::packed_largest);
    CHECK(at_largest.color.y == -contracts::packed_largest);
    CHECK(at_largest.color.z == contracts::packed_least_normal);
    CHECK(back.back().alpha == contracts::packed_absolute);
    CHECK(back.back().color.x == contracts::packed_absolute);
    CHECK(back.back().color.z == contracts::packed_absolute);
}

// ---- the light point -------------------------------------------------------

namespace {

const ProbeLight near_light{{0.0f, 2.0f, 0.0f}, 0.5f, 7.0f};

}  // namespace

TEST_CASE("a light point drawn from a point and evaluated there is the draw: direction, distance, radiance, density") {
    std::vector<tests::LightPointQuery> queries;
    constexpr int grid = 16;
    for (const contracts::Float3 point : {contracts::Float3{0.0f, 0.0f, 0.0f}, contracts::Float3{1.5f, 0.3f, -0.7f},
                                          contracts::Float3{0.0f, 2.0f, 0.6f}}) {
        for (int i = 0; i < grid * grid; ++i) {
            queries.push_back({point, (static_cast<float>(i % grid) + 0.5f) / grid,
                               (static_cast<float>(i / grid) + 0.5f) / grid, {0u, 0u, 0u}});
        }
    }
    // One inside the light: no sample, as sample_light() draws none.
    queries.push_back({{0.0f, 2.1f, 0.0f}, 0.5f, 0.5f, {0u, 0u, 0u}});
    const auto count = static_cast<std::uint32_t>(queries.size());
    tests::ProbeRunner gpu;
    const std::vector<tests::LightPointResult> results = run_with_light<tests::LightPointResult>(
        gpu, "light_points", count, near_light, {Binding::of(queries), Binding::of(count)});

    const double radius = near_light.radius;
    for (std::size_t i = 0; i + 1 < results.size(); ++i) {
        const tests::LightPointResult& r = results[i];
        INFO("query " << i);
        REQUIRE(contracts::holds_light(contracts::LightPoint{r.light, 0u}));
        CHECK(contracts::light_index(contracts::LightPoint{r.light, 0u}) == 0u);
        // The point is its cell's: within the round trip's angle on the unit
        // sphere, r times it in the world, seen from the point at its
        // distance.
        const double moved = radius * contracts::unit_round_trip_error;
        CHECK(angle_between(tests::vec(r.direction), tests::vec(r.sample_direction)) <=
              1.01 * moved / r.sample_distance + 1e-6);
        CHECK(std::abs(r.distance - r.sample_distance) <= 1.01 * moved + 1e-6);
        CHECK(r.radiance.x == r.sample_radiance.x);
        CHECK(r.radiance.y == r.sample_radiance.y);
        CHECK(r.pdf == doctest::Approx(r.sample_pdf).epsilon(1e-6));
        CHECK(r.area_scale == doctest::Approx(radius * radius).epsilon(1e-6));
        CHECK(r.cos_light > 0.0f);
    }
    CHECK_FALSE(contracts::holds_light(contracts::LightPoint{results.back().light, 0u}));
    CHECK(results.back().sample_pdf == 0.0f);
}

TEST_CASE("a light point's density in the light's own area is the closed form's, and integrates its cap") {
    // From a point at distance D from a sphere of radius r, the cone is
    // drawn uniformly over its solid angle Omega = 2 pi (1 - sqrt(1 -
    // r^2 / D^2)), so the density per unit of the unit sphere's area at a
    // point y is cos(theta_y) r^2 / (|y - x|^2 Omega). And over the cap the
    // point can see, where it can be drawn, the mean of 1 / density is that
    // cap's area on the unit sphere, 2 pi (1 - r / D): the conversion
    // integrates to one over the area it converts into. 1 / density grows
    // as (1 - u0)^(-1/2) toward the cap's rim (u0 -> 1), which a grid in u0
    // under-counts; the grid is in s, u0 = 1 - s^2, each draw weighed by
    // the substitution's 2 s, which makes the integrand smooth.
    constexpr int rows = 256;
    constexpr int columns = 64;
    const contracts::Float3 at{0.4f, -0.5f, 0.3f};
    std::vector<tests::LightPointQuery> queries;
    std::vector<double> substitution;
    for (int i = 0; i < rows * columns; ++i) {
        const double s = (i / columns + 0.5) / rows;
        substitution.push_back(2.0 * s);
        queries.push_back({at, static_cast<float>(1.0 - s * s), (static_cast<float>(i % columns) + 0.5f) / columns,
                           {0u, 0u, 0u}});
    }
    const auto count = static_cast<std::uint32_t>(queries.size());
    tests::ProbeRunner gpu;
    const std::vector<tests::LightPointResult> results = run_with_light<tests::LightPointResult>(
        gpu, "light_points", count, near_light, {Binding::of(queries), Binding::of(count)});

    const Vec3 x = tests::vec(at);
    const Vec3 center = tests::vec(near_light.center);
    const double radius = near_light.radius;
    const double d = tests::length(center - x);
    const double omega = 2.0 * std::numbers::pi * (1.0 - std::sqrt(1.0 - radius * radius / (d * d)));
    double inverse_sum = 0.0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const tests::LightPointResult& r = results[i];
        const Vec3 y = x + static_cast<double>(r.distance) * tests::vec(r.direction);
        const Vec3 outward = tests::normalized(y - center);
        const double cos_y = -dot(outward, tests::vec(r.direction));
        const double expected = cos_y * radius * radius / (r.distance * r.distance * omega);
        CHECK(r.area_density == doctest::Approx(expected).epsilon(1e-4));
        CHECK(r.cos_light == doctest::Approx(cos_y).epsilon(1e-4));
        inverse_sum += substitution[i] / r.area_density;
    }
    const double cap = 2.0 * std::numbers::pi * (1.0 - radius / d);
    MESSAGE("mean of 1 / density " << inverse_sum / count << ", the cap's area " << cap);
    CHECK(inverse_sum / count == doctest::Approx(cap).epsilon(0.01));
}

// ---- the target ------------------------------------------------------------

TEST_CASE("the target, its candidate weight and the direct light: closed forms for a Lambert floor under one light") {
    // A white-light firefly over a Lambert floor (normal +y). At floor
    // point x, for the point y on the light the identity holds, the light
    // gives L along y - x, and
    //   f = albedo / pi,  cos_x = the direction's y,  cos_y at the light,
    //   target = luminance(f cos_x L) cos_y r^2 / d^2 (luminance of a grey
    //            is its value: Rec. 709's weights sum to 1),
    //   weight = luminance(f cos_x L) / (P / Omega), the cancelled p̂ / p,
    //   direct = f cos_x L cos_y r^2 / d^2 (W = 1), every channel,
    //   density = cos_y r^2 / (d^2 Omega).
    // y is the decoded point of the direction from the light's center
    // toward x (its nearest point, cos_y near 1), and of the opposite one,
    // on the far side, which gives nothing.
    const ProbeLight firefly{{0.3f, 1.5f, -0.2f}, 0.05f, 200.0f};
    constexpr double albedo = 0.8;
    constexpr double selection = 0.25;  // as if one of four lights
    const std::vector<contracts::Float3> floor_points{
        {0.0f, 0.0f, 0.0f}, {1.2f, 0.0f, 0.4f}, {-0.8f, 0.0f, -1.1f}, {0.3f, 0.0f, -0.2f}};
    std::vector<contracts::Float3> toward;
    for (const contracts::Float3& p : floor_points) {
        const Vec3 v = tests::normalized(tests::vec(p) - tests::vec(firefly.center));
        toward.push_back({static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)});
        toward.push_back({static_cast<float>(-v.x), static_cast<float>(-v.y), static_cast<float>(-v.z)});
    }
    tests::ProbeRunner gpu;
    const auto trip_count = static_cast<std::uint32_t>(toward.size());
    const std::vector<tests::UnitTrip> trips =
        gpu.run<tests::UnitTrip>("unit_trip", trip_count, {Binding::of(toward), Binding::of(trip_count)});

    std::vector<tests::TargetQuery> queries;
    for (std::size_t i = 0; i < toward.size(); ++i) {
        queries.push_back({floor_points[i / 2], trips[i].code, {0.0f, 1.0f, 0.0f}, static_cast<float>(selection)});
    }
    contracts::Bsdf floor{};
    floor.normal = {0.0f, 1.0f, 0.0f};
    floor.kind = contracts::BsdfKind::lambert;
    floor.color = {0.8f, 0.8f, 0.8f};
    const auto count = static_cast<std::uint32_t>(queries.size());
    const std::vector<tests::TargetResult> results = run_with_light<tests::TargetResult>(
        gpu, "target_at", count, firefly, {Binding::of(queries), Binding::of(count), Binding::of(floor)});

    const Vec3 center = tests::vec(firefly.center);
    const double radius = firefly.radius;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const tests::TargetResult& r = results[i];
        const Vec3 x = tests::vec(floor_points[i / 2]);
        const Vec3 on_unit = tests::vec(trips[i].decoded);
        const Vec3 y = center + radius * on_unit;
        const double d = tests::length(y - x);
        const Vec3 wi = (1.0 / d) * (y - x);
        const double cos_x = wi.y;
        const double cos_y = -dot(on_unit, wi);
        const double big_d = tests::length(center - x);
        const double omega = 2.0 * std::numbers::pi * (1.0 - std::sqrt(1.0 - radius * radius / (big_d * big_d)));
        const double scattered = albedo / std::numbers::pi * cos_x * firefly.radiance;
        INFO("query " << i << ": cos_y " << cos_y);
        if (i % 2 == 1) {
            // The far side: nothing.
            CHECK(cos_y < 0.0);
            CHECK(r.target == 0.0f);
            CHECK(r.weight == 0.0f);
            CHECK(r.area_density == 0.0f);
            CHECK(r.direct.x == 0.0f);
            continue;
        }
        const double g = cos_y * radius * radius / (d * d);
        CHECK(r.distance == doctest::Approx(d).epsilon(1e-5));
        CHECK(r.target == doctest::Approx(scattered * g).epsilon(1e-4));
        CHECK(r.weight == doctest::Approx(scattered / (selection / omega)).epsilon(1e-4));
        CHECK(r.area_density == doctest::Approx(g / omega).epsilon(1e-4));
        CHECK(r.direct.x == doctest::Approx(scattered * g).epsilon(1e-4));
        CHECK(r.direct.z == doctest::Approx(scattered * g).epsilon(1e-4));
        // The weight is the target over the density the candidate was drawn
        // with, selection times the area density: p̂ / p.
        CHECK(r.weight == doctest::Approx(r.target / (selection * r.area_density)).epsilon(1e-4));
    }
}

// ---- the reservoir against softmax -----------------------------------------

namespace {

// A row of attention scores (docs/process/MEMORY.md: the reservoir merge
// against softmax).
constexpr std::array<float, 8> scores{0.3f, -1.2f, 2.0f, 0.0f, 1.1f, -0.5f, 0.7f, -2.0f};
constexpr std::uint32_t streams = 1u << 20;

tests::SoftmaxQuery softmax_query(std::uint32_t order, float scale, std::uint32_t seed) {
    tests::SoftmaxQuery q{};
    q.count = static_cast<std::uint32_t>(scores.size());
    q.streams = streams;
    q.order = order;
    q.seed = seed;
    q.scale = scale;
    std::copy(scores.begin(), scores.end(), q.scores);
    return q;
}

// How often each key was kept.
std::vector<double> kept_counts(const std::vector<tests::SoftmaxDraw>& draws) {
    std::vector<double> counted(scores.size(), 0.0);
    for (const tests::SoftmaxDraw& d : draws) {
        REQUIRE(contracts::holds_light(contracts::LightPoint{d.light, 0u}));
        counted.at(contracts::light_index(contracts::LightPoint{d.light, 0u})) += 1.0;
    }
    return counted;
}

// softmax(scores) x streams.
std::vector<double> softmax_counts() {
    double sum = 0.0;
    for (const float s : scores) {
        sum += std::exp(static_cast<double>(s));
    }
    std::vector<double> expected;
    for (const float s : scores) {
        expected.push_back(std::exp(static_cast<double>(s)) / sum * streams);
    }
    return expected;
}

}  // namespace

TEST_CASE("a reservoir fed exp(score) keeps key i with probability softmax(score)_i, in any order of the stream") {
    tests::ProbeRunner gpu;
    const std::vector<double> expected = softmax_counts();
    std::vector<std::vector<double>> by_order;
    for (const std::uint32_t order : {tests::order_forward, tests::order_reversed, tests::order_rotated}) {
        const std::vector<tests::SoftmaxDraw> draws =
            gpu.run<tests::SoftmaxDraw>("softmax_streams", streams, {Binding::of(softmax_query(order, 1.0f, 17u))});
        by_order.push_back(kept_counts(draws));
        const ChiSquare fit = chi_square(by_order.back(), expected);
        MESSAGE("order " << order << ": chi^2 " << fit.statistic << ", " << fit.dof << " dof, p " << fit.p);
        CHECK(fit.p > significance);
        // Finished as resampled importance sampling: W = w_sum / (M p̂), so
        // W M p̂ is the stream's sum, the same for every stream.
        for (std::size_t i = 0; i < draws.size(); i += 4099) {
            const tests::SoftmaxDraw& d = draws[i];
            CHECK(d.count == scores.size());
            CHECK(static_cast<double>(d.weight) * d.count * d.target == doctest::Approx(d.weight_sum).epsilon(1e-5));
        }
    }
    // The orders against one another, a 2 x 8 table each against the
    // forward one: the same distribution.
    for (std::size_t o = 1; o < by_order.size(); ++o) {
        double statistic = 0.0;
        for (std::size_t k = 0; k < scores.size(); ++k) {
            const double pooled = 0.5 * (by_order[0][k] + by_order[o][k]);
            statistic += (by_order[0][k] - pooled) * (by_order[0][k] - pooled) / pooled +
                         (by_order[o][k] - pooled) * (by_order[o][k] - pooled) / pooled;
        }
        const double p = upper_gamma(0.5 * (scores.size() - 1), 0.5 * statistic);
        MESSAGE("forward against order " << o << ": chi^2 " << statistic << ", p " << p);
        CHECK(p > significance);
    }
}

TEST_CASE("the softmax test can fail: weights exp(2 score) are refused") {
    tests::ProbeRunner gpu;
    const std::vector<tests::SoftmaxDraw> draws =
        gpu.run<tests::SoftmaxDraw>("softmax_streams", streams, {Binding::of(softmax_query(tests::order_forward, 2.0f, 17u))});
    const ChiSquare fit = chi_square(kept_counts(draws), softmax_counts());
    MESSAGE("exp(2 score) against softmax(score): chi^2 " << fit.statistic << ", " << fit.dof << " dof, p " << fit.p);
    CHECK(fit.p < 1e-12);
}

TEST_CASE("a stream of weights 0 keeps nothing: no sample, W = 0, its count kept") {
    tests::SoftmaxQuery q = softmax_query(tests::order_forward, 1.0f, 3u);
    q.zero = 1u;
    q.streams = 64u;
    tests::ProbeRunner gpu;
    for (const tests::SoftmaxDraw& d : gpu.run<tests::SoftmaxDraw>("softmax_streams", 64u, {Binding::of(q)})) {
        CHECK_FALSE(contracts::holds_light(contracts::LightPoint{d.light, 0u}));
        CHECK(d.weight == 0.0f);
        CHECK(d.target == 0.0f);
        CHECK(d.count == scores.size());
    }
}

// ---- the shade's rule for a reservoir's test -------------------------------

TEST_CASE("the shade takes a reservoir's own test where it is here and now, or recent and near in basic mode") {
    const auto word = [](int x, int y, std::uint32_t age) {
        return (static_cast<std::uint32_t>(x) & 0xffu) | ((static_cast<std::uint32_t>(y) & 0xffu) << 8u) |
               (age << 16u);
    };
    const std::vector<std::uint32_t> tested{word(0, 0, 0), word(0, 0, 1),  word(-3, 4, 4), word(-3, 4, 5),
                                            word(16, 0, 1), word(15, 0, 1), word(0, -15, 0), word(-127, -127, 255)};
    const auto count = static_cast<std::uint32_t>(tested.size());
    tests::ProbeRunner gpu;
    const auto taken = [&](const passes::RestirDi& settings) {
        return gpu.run<std::uint32_t>("takes_test_of", count,
                                      {Binding::of(tested), Binding::of(count), Binding::of(settings)});
    };
    passes::RestirDi ris{};  // no merge: nothing but here and now
    ris.candidates = 8;
    CHECK(taken(ris) == std::vector<std::uint32_t>{1, 0, 0, 0, 0, 0, 0, 0});
    passes::RestirDi basic = ris;
    basic.history = 20;
    basic.bias_correction = contracts::BiasCorrection::basic;
    basic.visibility_reuse = contracts::VisibilityReuse::on;
    basic.visibility_age = 4;
    basic.visibility_distance = 16.0f;
    // Age at most 4, offset shorter than 16 pixels.
    CHECK(taken(basic) == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 1, 1, 0});
    passes::RestirDi raytraced = basic;
    raytraced.bias_correction = contracts::BiasCorrection::raytraced;
    CHECK(taken(raytraced) == std::vector<std::uint32_t>{1, 0, 0, 0, 0, 0, 0, 0});
}

// ---- the renderer ----------------------------------------------------------

namespace {

// A graph of resampled importance sampling with `candidates`, its
// accumulated image shown as it is.
frame::Schedule ris_graph(std::uint32_t candidates) {
    return frame::parse_schedule("passes = [\"restir_initial\", \"restir_shade\", \"display\"]\n[restir_di]\n"
                                 "candidates = " + std::to_string(candidates) + "\n",
                                 "ris graph");
}

// A renderer of `schedule` over a scene, at a size, frames accumulating
// from frame 0, read back in the accumulated image's floats (C.8: it holds
// what the renderer reads, so it is neither copied nor moved).
class Rig {
public:
    Rig(const std::string& text, const frame::Schedule& schedule, frame::Extent image)
        : description_(scene::parse(text, "test scene")), size_(image), renderer_(device_, submission_, schedule, &description_) {}

    // Renders `frames` more frames and returns the accumulated image, RGBA
    // floats, every sample of them finite.
    std::vector<float> render(std::uint64_t frames) {
        std::uint64_t sequence = 0;
        for (std::uint64_t i = 0; i < frames; ++i, ++next_) {
            sequence = metal::render_to_offscreen(submission_, target_, renderer_,
                                                  tests::frame_at(next_, 0, 0.0, description_.camera));
        }
        (void)submission_.wait_until_complete(sequence);
        CHECK(renderer_.non_finite_samples() == 0);
        std::vector<float> floats(std::size_t{size_.width} * size_.height * 4);
        renderer_.read_accumulated(floats);
        return floats;
    }

    const scene::SceneDescription& description() const noexcept { return description_; }
    frame::Extent size() const noexcept { return size_; }

private:
    scene::SceneDescription description_;
    frame::Extent size_;
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Offscreen target_{device_, submission_, size_};
    metal::Renderer renderer_;
    std::uint64_t next_ = 0;
};

// The red mean over the 3 x 3 pixels about `at`.
double patch_red(const std::vector<float>& rgba, frame::Extent size, tests::Pixel at) {
    double sum = 0.0;
    for (std::uint32_t y = at.y - 1; y <= at.y + 1; ++y) {
        for (std::uint32_t x = at.x - 1; x <= at.x + 1; ++x) {
            sum += rgba[(std::size_t{y} * size.width + x) * 4];
        }
    }
    return sum / 9.0;
}

constexpr double floor_albedo = 0.8;

// The lit floor of path_test.cpp: a Lambert floor in a black sky under
// fireflies of radius 0.05 and radiance 200, seen from above.
std::string lit_floor(const std::string& rest) {
    return tests::camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + tests::sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
           "[materials.black]\nkind = \"rough\"\ncolor = [0, 0, 0]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n" +
           rest;
}

std::string firefly(const char* center) {
    return std::string("[[shapes]]\nkind = \"sphere\"\ncenter = ") + center + "\nradius = 0.05\nmaterial = \"glow\"\n";
}

const std::string one_firefly = firefly("[0.3, 2.5, -0.5]");
const std::string two_fireflies = one_firefly + firefly("[-1.2, 1.5, 0.4]");

// The floor point pixel (x, y)'s center sees.
Vec3 floor_point(const scene::SceneDescription& description, frame::Extent size, tests::Pixel at) {
    const contracts::CameraData c = camera::shader_form(description.camera, size);
    const Vec3 d = tests::camera_direction(c, at.x + 0.5, at.y + 0.5, size);
    return tests::vec(c.origin) + (-c.origin.y / d.y) * d;
}

// albedo x L sin^2 cos for each of `lights` (their indices among the
// scene's sphere lights), at floor point p: what an unshadowed firefly
// gives the floor.
double floor_from(const scene::SceneDescription& description, Vec3 p, const std::vector<std::size_t>& lights) {
    double sum = 0.0;
    for (const std::size_t i : lights) {
        const lights::SphereLightData& light = description.sphere_lights.at(i);
        const contracts::Transform& placed = description.shapes.transforms.at(light.shape);
        const double radius = placed.m[0][0];
        const Vec3 to = tests::vec(contracts::translation(placed)) - p;
        const double d2 = dot(to, to);
        sum += floor_albedo * light.radiance.x * (radius * radius / d2) * (to.y / std::sqrt(d2));
    }
    return sum;
}

std::vector<std::size_t> every_light(const scene::SceneDescription& description) {
    std::vector<std::size_t> all(description.sphere_lights.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
        all[i] = i;
    }
    return all;
}

constexpr frame::Extent floor_size{64, 48};
constexpr std::array<tests::Pixel, 3> floor_pixels{tests::Pixel{32, 30}, tests::Pixel{14, 38}, tests::Pixel{50, 26}};

// Every floor pixel's patch against the closed form, within `tolerance`.
void check_floor(const std::string& text, std::uint32_t candidates, std::uint64_t frames, double tolerance) {
    Rig rig(text, ris_graph(candidates), floor_size);
    const std::vector<float> image = rig.render(frames);
    for (const tests::Pixel pixel : floor_pixels) {
        const double expected =
            floor_from(rig.description(), floor_point(rig.description(), rig.size(), pixel), every_light(rig.description()));
        INFO("M = " << candidates << ", pixel " << pixel.x << ", " << pixel.y << ": expected " << expected);
        CHECK(patch_red(image, rig.size(), pixel) == doctest::Approx(expected).scale(0).epsilon(tolerance));
    }
}

// The root mean square of the floor pixels' relative error, every pixel
// of the floor's lower half (all floor), after `frames`.
double floor_rms(const std::string& text, std::uint32_t candidates, std::uint64_t frames) {
    Rig rig(text, ris_graph(candidates), floor_size);
    const std::vector<float> image = rig.render(frames);
    double sum = 0.0;
    std::size_t n = 0;
    for (std::uint32_t y = floor_size.height / 2; y < floor_size.height; ++y) {
        for (std::uint32_t x = 0; x < floor_size.width; ++x) {
            const double expected = floor_from(rig.description(), floor_point(rig.description(), rig.size(), {x, y}),
                                               every_light(rig.description()));
            const double e = image[(std::size_t{y} * floor_size.width + x) * 4] / expected - 1.0;
            sum += e * e;
            ++n;
        }
    }
    return std::sqrt(sum / static_cast<double>(n));
}

}  // namespace

TEST_CASE("resampled importance sampling: a floor under one firefly converges to albedo L sin^2 cos") {
    check_floor(lit_floor(one_firefly), 8, 512, 0.03);
}

TEST_CASE("resampled importance sampling with M = 1 is next event estimation: under two fireflies, their sum") {
    // One candidate, kept: W = 1 / p, the naive estimator's weight; the
    // path tracer's floor test (path_test.cpp) with the same scene and
    // tolerance.
    check_floor(lit_floor(two_fireflies), 1, 1024, 0.03);
}

TEST_CASE("resampled importance sampling with M = 8 converges to the same mean under two fireflies, with less noise") {
    check_floor(lit_floor(two_fireflies), 8, 1024, 0.03);
    // At equal frames, eight candidates weighed by the light they give
    // choose the firefly that matters more often than one chosen blind.
    const double one = floor_rms(lit_floor(two_fireflies), 1, 16);
    const double eight = floor_rms(lit_floor(two_fireflies), 8, 16);
    MESSAGE("relative RMS error after 16 frames: M = 1 " << one << ", M = 8 " << eight);
    CHECK(eight < 0.75 * one);
}

TEST_CASE("visibility reuse: a light hidden behind a box gives nothing, a visible one its whole light") {
    // A black box held over the floor, a firefly straight above it, and
    // another low to the side, under the box's edge. The floor beneath the
    // box sees only the low one; its light is all it shows, and with the
    // low one gone, it shows nothing, to the last bit. The box is black and
    // the sky too, so nothing bounces light there: the direct light is the
    // whole of it. The two give the floor beneath about the same light, so
    // its candidates keep each about as often: a frame that keeps the
    // hidden one shows nothing there (W = 0), and the mean over frames is
    // the low one's light, unbiased, with the noise of a coin's toss, which
    // 2048 frames hold to a percent or so.
    const std::string box = "[[shapes]]\nkind = \"box\"\nmin = [-0.6, 0.6, -0.6]\nmax = [0.6, 0.7, 0.6]\n"
                            "material = \"black\"\n";
    const std::string above = firefly("[0, 4, 0]");
    const std::string low = firefly("[2, 0.6, 0]");
    const std::string camera = tests::camera_text("[0, 0.35, 2.5]", "[0, 0.05, 0]", 40);
    const auto scene_of = [&](const std::string& lights) {
        return camera + tests::sky("[0, 0, 0]", "[0, 0, 0]") +
               "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
               "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
               "[materials.black]\nkind = \"rough\"\ncolor = [0, 0, 0]\n"
               "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n" +
               box + lights;
    };
    constexpr frame::Extent size{64, 48};

    // The floor beneath the box's middle, where the box hides the firefly
    // above it whole, and the floor beside the box, which sees both.
    Rig both(scene_of(above + low), ris_graph(8), size);
    const contracts::CameraData c = camera::shader_form(both.description().camera, size);
    const tests::Pixel under = tests::pixel_of(c, Vec3{0.0, 0.0, 0.1}, size);
    const tests::Pixel beside = tests::pixel_of(c, Vec3{0.4, 0.0, 1.4}, size);
    const std::vector<float> lit = both.render(2048);
    const Vec3 under_point = floor_point(both.description(), size, under);
    REQUIRE(std::abs(under_point.x) < 0.4);
    REQUIRE(std::abs(under_point.z) < 0.4);
    const double from_low = floor_from(both.description(), under_point, {1});
    INFO("beneath the box: expected " << from_low << " from the low firefly alone");
    CHECK(patch_red(lit, size, under) == doctest::Approx(from_low).scale(0).epsilon(0.03));
    const double from_both = floor_from(both.description(), floor_point(both.description(), size, beside), {0, 1});
    CHECK(patch_red(lit, size, beside) == doctest::Approx(from_both).scale(0).epsilon(0.03));

    Rig hidden(scene_of(above), ris_graph(8), size);
    const std::vector<float> dark = hidden.render(64);
    CHECK(patch_red(dark, size, under) == 0.0);
    CHECK(patch_red(dark, size, beside) > 0.0);
}

TEST_CASE("resampled importance sampling: the same inputs give the same image, to the bit") {
    Rig first(lit_floor(two_fireflies), ris_graph(8), {32, 24});
    Rig second(lit_floor(two_fireflies), ris_graph(8), {32, 24});
    const std::vector<float> a = first.render(4);
    const std::vector<float> b = second.render(4);
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    // And the comparison can fail: one frame more is another image.
    CHECK(std::memcmp(a.data(), second.render(1).data(), a.size() * sizeof(float)) != 0);
}

TEST_CASE("the Metal backend refuses ReSTIR's merges by name until it implements them") {
    const scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    const std::string merges =
        "[restir_di]\ncandidates = 8\nhistory = 20\npermutation = true\nneighbors = 1\nradius = 32.0\n"
        "depth_threshold = 0.1\nnormal_threshold = 0.5\nbias_correction = \"basic\"\nvisibility_reuse = true\n"
        "visibility_age = 4\nvisibility_distance = 16.0\n";
    const auto refusal = [&](const char* passes, const std::string& settings) {
        const frame::Schedule schedule = frame::parse_schedule(std::string("passes = ") + passes + "\n" + settings, "g");
        return tests::error_of<metal::MetalError>([&] { return metal::Renderer(device, submission, schedule, &description); });
    };
    const std::string temporal_only = tests::replaced(
        tests::replaced(merges, "neighbors = 1\n", ""), "radius = 32.0\n", "");
    CHECK(tests::contains(refusal("[\"restir_initial\", \"restir_temporal\", \"restir_shade\", \"display\"]",
                                  temporal_only),
                          "restir_temporal pass is not implemented by the Metal backend"));
    const std::string spatial_only =
        tests::replaced(tests::replaced(merges, "history = 20\n", ""), "permutation = true\n", "");
    CHECK(tests::contains(refusal("[\"restir_initial\", \"restir_spatial\", \"restir_shade\", \"display\"]",
                                  spatial_only),
                          "restir_spatial pass is not implemented by the Metal backend"));
    // The graph without them is built.
    CHECK(refusal("[\"restir_initial\", \"restir_shade\", \"display\"]", "[restir_di]\ncandidates = 8\n").empty());
}

TEST_CASE("a ReSTIR graph refuses a scene with more media than its path state's 16 bits hold") {
    scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    const auto refusal = [&](const frame::Schedule& schedule) {
        return tests::error_of<metal::MetalError>([&] { return metal::Renderer(device, submission, schedule, &description); });
    };
    description.media.resize(0xfffe);  // the most it holds: 0xffff is none
    CHECK(refusal(ris_graph(8)).empty());
    description.media.resize(0xffff);
    CHECK(tests::contains(refusal(ris_graph(8)), "keeps a medium's index in 16 bits"));
    // The path graph keeps no path state, and takes it.
    CHECK(refusal(tests::path_graph()).empty());
}

TEST_CASE("a ReSTIR graph refuses a scene whose materials its packed surface record cannot hold, by name") {
    metal::Device device;
    metal::Submission submission(device);
    const auto refusal = [&](const std::string& extra, const frame::Schedule& schedule) {
        const scene::SceneDescription description = scene::parse(lit_floor(one_firefly + extra), "test scene");
        return tests::error_of<metal::MetalError>(
            [&] { return metal::Renderer(device, submission, schedule, &description); });
    };
    const auto glass = [](const char* ior) {
        return std::string("[materials.glass]\nkind = \"dielectric\"\nior = ") + ior +
               "\n[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.3, 0]\nradius = 0.3\nmaterial = \"glass\"\n";
    };
    // Half's largest is held (contracts/bsdf.h); past it, refused naming the
    // material and the value; the path graph, which packs nothing, takes it.
    CHECK(refusal(glass("65504"), ris_graph(8)).empty());
    const std::string refused = refusal(glass("70000"), ris_graph(8));
    CHECK(tests::contains(refused, "material 'glass''s ior, 70000, is outside what ReSTIR DI's packed surface record"));
    CHECK(refusal(glass("70000"), tests::path_graph()).empty());
}

TEST_CASE("ReSTIR's arrays: what the merges need is refused until they come, and a size Metal cannot make") {
    metal::Device device;
    metal::Submission submission(device);
    using metal::ReuseImages;
    CHECK(tests::contains(tests::error_of<metal::MetalError>([&] {
                              return ReuseImages(device, submission, ReuseImages::Temporal::merged,
                                                 ReuseImages::Working::none);
                          }),
                          "restir_temporal's, which this backend does not implement yet"));
    CHECK(tests::contains(tests::error_of<metal::MetalError>([&] {
                              return ReuseImages(device, submission, ReuseImages::Temporal::none,
                                                 ReuseImages::Working::needed);
                          }),
                          "the working reservoirs are the merges'"));
    ReuseImages arrays(device, submission, ReuseImages::Temporal::none, ReuseImages::Working::none);
    CHECK(arrays.addresses(0).records == 0u);  // nothing before the first prepare()
    const frame::FrameInputs inputs = tests::frame_at(0, 0, 0.0, std::nullopt);
    CHECK_THROWS_AS((void)arrays.prepare(inputs, {0, 4}), metal::MetalError);
    CHECK(arrays.prepare(inputs, {8, 4}) == 0u);  // no frame before to merge
    const ReuseImages::Addresses made = arrays.addresses(0);
    CHECK(made.records != 0u);
    CHECK(made.path_states != 0u);
    CHECK(made.final_reservoirs != 0u);
    // The merges' arrays are not made.
    CHECK(made.last_records == 0u);
    CHECK(made.previous_positions == 0u);
    CHECK(made.working == 0u);
    CHECK(made.last_final == 0u);
}

TEST_CASE("ReSTIR's passes refuse a frame without what they read and write") {
    metal::Device device;
    const metal::Library library(device, serenity::metallib::shaders);
    // Addresses never dispatched: each record() refuses before it records.
    constexpr MTL::GPUAddress somewhere = 0x1000;
    metal::FrameResources full{};
    full.scene = somewhere;
    full.transforms = somewhere;
    full.glows = somewhere;
    full.camera = somewhere;
    full.reuse = somewhere;
    full.records = somewhere;
    full.path_states = somewhere;
    full.final_reservoirs = somewhere;
    full.non_finite_counter = somewhere;
    const auto refusal = [](const auto& pass, const metal::FrameResources& resources) {
        return tests::error_of<metal::MetalError>([&] {
            pass.record(nullptr, resources);
            return 0;
        });
    };
    const metal::RestirInitialPass initial(library, metal::ReservoirOutput::final);
    CHECK(tests::contains(refusal(initial, metal::FrameResources{}), "RestirInitialPass: the frame lacks"));
    metal::FrameResources no_final = full;
    no_final.final_reservoirs = 0;
    CHECK(tests::contains(refusal(initial, no_final), "no final reservoirs to write"));
    const metal::RestirInitialPass into_working(library, metal::ReservoirOutput::working);
    CHECK(tests::contains(refusal(into_working, full), "no working reservoirs to write"));
    metal::FrameResources temporal = full;
    temporal.previous_positions = somewhere;
    CHECK(tests::contains(refusal(initial, temporal), "previous positions are written for restir_temporal"));

    const metal::RestirShadePass shade(library);
    CHECK(tests::contains(refusal(shade, metal::FrameResources{}), "RestirShadePass: the frame lacks"));
    // The shade needs the accumulated image too.
    CHECK(tests::contains(refusal(shade, full), "RestirShadePass: the frame lacks"));
}
