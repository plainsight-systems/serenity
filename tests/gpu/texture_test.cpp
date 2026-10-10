// The procedural textures on the GPU (tests/gpu/kernels/texture_probe.metal),
// held to what their headers promise: the noise (core/textures/noise.h) 0 at
// every lattice point, within its bound, continuous, centered on 0, and a
// function of the point and the seed; the wood (core/textures/wood.h) within
// its bounds and finite across the world, the same at every height, dark at
// the seams, and different from board to board and seed to seed; the swirl
// (core/textures/swirl.h) between its colors, the same at any distance from
// its axis, and turning with its twist.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/textures/noise.h"
#include "core/textures/swirl.h"
#include "core/textures/wood.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"

using namespace serenity;
using tests::Binding;
using tests::TexturePoint;

namespace {

double uniform(std::uint64_t i, std::uint64_t which) {
    return animation::draw(0x5EEDu, i, which);
}

TexturePoint at(double x, double y, double z, std::uint32_t seed = 0) {
    return {{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}, seed};
}

std::vector<float> noise(tests::ProbeRunner& gpu, const std::vector<TexturePoint>& points) {
    const auto count = static_cast<std::uint32_t>(points.size());
    return gpu.run<float>("noise_probe", points.size(), {Binding::of(points), Binding::of(count)});
}

// The colors a texture of `data` gives at `points`, by `kernel`.
template <typename Data>
std::vector<contracts::Float3> colors(tests::ProbeRunner& gpu, const char* kernel,
                                      const std::vector<TexturePoint>& points, const Data& data) {
    const auto count = static_cast<std::uint32_t>(points.size());
    return gpu.run<contracts::Float3>(kernel, points.size(),
                                      {Binding::of(points), Binding::of(count), Binding::of(data)});
}

bool same(contracts::Float3 a, contracts::Float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// How many of two lists' values differ.
template <typename T, typename Same>
std::size_t differing(const std::vector<T>& a, const std::vector<T>& b, Same equal) {
    REQUIRE(a.size() == b.size());
    std::size_t differ = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        differ += equal(a[i], b[i]) ? 0u : 1u;
    }
    return differ;
}

constexpr textures::WoodData walnut{.light = {0.13f, 0.065f, 0.03f},
                                    .ring = 0.004f,
                                    .dark = {0.045f, 0.02f, 0.008f},
                                    .board = 0.16f,
                                    .seed = 3u,
                                    .padding = {0, 0, 0}};

}  // namespace

TEST_CASE("noise is 0 at every lattice point, whatever the seed") {
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    for (std::uint32_t i = 0; i < 4096; ++i) {
        const auto lattice = [&](std::uint64_t which) { return std::floor(-1000.0 + 2000.0 * uniform(i, which)); };
        points.push_back(at(lattice(0), lattice(1), lattice(2), i));
    }
    const std::vector<float> values = noise(gpu, points);
    CHECK(std::ranges::count_if(values, [](float v) { return v != 0.0f; }) == 0);
}

TEST_CASE("noise stays within its bound, centered on 0, and uses its range") {
    constexpr std::uint64_t count = std::uint64_t{1} << 20;
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    points.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        points.push_back(
            at(-50.0 + 100.0 * uniform(i, 0), -50.0 + 100.0 * uniform(i, 1), -50.0 + 100.0 * uniform(i, 2), 7u));
    }
    const std::vector<float> values = noise(gpu, points);
    double sum = 0.0;
    double largest = 0.0;
    for (const float v : values) {
        REQUIRE(std::isfinite(v));
        sum += v;
        largest = std::max(largest, static_cast<double>(std::abs(v)));
    }
    const double mean = sum / static_cast<double>(values.size());
    INFO("largest " << largest << ", mean " << mean);
    CHECK(largest <= textures::noise_bound);
    CHECK(largest > 0.6);
    CHECK(std::abs(mean) < 0.01);
}

TEST_CASE("noise is continuous across cells, and a function of the point and the seed") {
    // Pairs a micron apart, many straddling a cell's face.
    constexpr std::uint64_t pairs = 65536;
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    points.reserve(2 * pairs);
    for (std::uint64_t i = 0; i < pairs; ++i) {
        const double x = std::floor(-20.0 + 40.0 * uniform(i, 0)) + (i % 2 == 0 ? 0.0 : uniform(i, 3));
        const double y = -20.0 + 40.0 * uniform(i, 1);
        const double z = -20.0 + 40.0 * uniform(i, 2);
        points.push_back(at(x - 1e-4, y, z, 11u));
        points.push_back(at(x + 1e-4, y, z, 11u));
    }
    const std::vector<float> values = noise(gpu, points);
    double widest = 0.0;
    for (std::size_t i = 0; i < values.size(); i += 2) {
        widest = std::max(widest, static_cast<double>(std::abs(values[i] - values[i + 1])));
    }
    INFO("widest step over 0.2 mm of noise space: " << widest);
    CHECK(widest < 2e-3);

    // The same points again, the same values; another seed, others.
    CHECK(noise(gpu, points) == values);
    std::vector<TexturePoint> reseeded = points;
    for (TexturePoint& p : reseeded) {
        p.seed = 12u;
    }
    const auto equal = [](float a, float b) { return a == b; };
    CHECK(differing(noise(gpu, reseeded), values, equal) > values.size() * 9 / 10);
}

TEST_CASE("wood stays within its bounds and finite, everywhere in the world") {
    constexpr std::uint64_t count = std::uint64_t{1} << 18;
    const float lo = textures::wood_seam_shade * (1.0f - textures::wood_pores) * (1.0f - textures::wood_board_shade);
    const float hi = 1.0f + textures::wood_board_shade;
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    points.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        // Over the table, and anywhere within a million meters.
        const double reach = i % 2 == 0 ? 2.0 : 1.0e6;
        points.push_back(at(reach * (2.0 * uniform(i, 0) - 1.0), reach * (2.0 * uniform(i, 1) - 1.0),
                            reach * (2.0 * uniform(i, 2) - 1.0)));
    }
    std::size_t outside = 0;
    for (const contracts::Float3 c : colors(gpu, "wood_probe", points, walnut)) {
        for (int k = 0; k < 3; ++k) {
            const float v = contracts::component(c, k);
            const float light = contracts::component(walnut.light, k);
            const float dark = contracts::component(walnut.dark, k);
            REQUIRE(std::isfinite(v));
            const bool within = v >= lo * std::min(light, dark) * (1.0f - 1e-5f) &&
                                v <= hi * std::max(light, dark) * (1.0f + 1e-5f);
            outside += within ? 0u : 1u;
        }
    }
    CHECK(outside == 0);
}

TEST_CASE("wood is the same at every height, dark at the seams, and differs board to board and seed to seed") {
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    for (std::uint64_t i = 0; i < 4096; ++i) {
        const double x = -1.4 + 2.8 * uniform(i, 0);
        const double z = -0.9 + 2.6 * uniform(i, 1);
        points.push_back(at(x, 0.75, z));
        points.push_back(at(x, -3.0 + 6.0 * uniform(i, 2), z));
    }
    const std::vector<contracts::Float3> grain = colors(gpu, "wood_probe", points, walnut);
    std::size_t by_height = 0;
    for (std::size_t i = 0; i < grain.size(); i += 2) {
        by_height += same(grain[i], grain[i + 1]) ? 0u : 1u;
    }
    CHECK(by_height == 0);

    // Across board 2's seam with board 3, at 3 boards: a millimetre either
    // side is seam, 3 mm is not.
    const double seam_at = 3.0 * walnut.board;
    std::vector<TexturePoint> seam;
    for (std::uint64_t i = 0; i < 512; ++i) {
        const double x = -1.4 + 2.8 * uniform(i, 3);
        for (const double dz : {-0.003, -0.001, 0.001, 0.003}) {
            seam.push_back(at(x, 0.75, seam_at + dz));
        }
    }
    const std::vector<contracts::Float3> across = colors(gpu, "wood_probe", seam, walnut);
    double in_seam = 0.0;
    double beside = 0.0;
    for (std::size_t i = 0; i < across.size(); i += 4) {
        in_seam += across[i + 1].x + across[i + 2].x;
        beside += across[i].x + across[i + 3].x;
    }
    CHECK(in_seam < 0.4 * beside);

    // Each board's mean differs from its neighbor's: each from its own log.
    constexpr std::size_t boards = 8;
    constexpr std::size_t per_board = 2048;
    std::vector<TexturePoint> planks_at;
    planks_at.reserve(boards * per_board);
    for (std::size_t b = 0; b < boards; ++b) {
        for (std::uint64_t i = 0; i < per_board; ++i) {
            planks_at.push_back(at(-1.4 + 2.8 * uniform(i, 4), 0.75,
                                   walnut.board * (static_cast<double>(b) + 0.05 + 0.9 * uniform(i, 5))));
        }
    }
    const std::vector<contracts::Float3> planks = colors(gpu, "wood_probe", planks_at, walnut);
    std::array<double, boards> means{};
    for (std::size_t b = 0; b < boards; ++b) {
        for (std::size_t i = 0; i < per_board; ++i) {
            means[b] += planks[b * per_board + i].x / static_cast<double>(per_board);
        }
    }
    for (std::size_t b = 1; b < boards; ++b) {
        INFO("boards " << b - 1 << " and " << b << ": " << means[b - 1] << ", " << means[b]);
        CHECK(std::abs(means[b] - means[b - 1]) > 1e-4);
    }

    textures::WoodData other = walnut;
    other.seed = 4u;
    CHECK(differing(colors(gpu, "wood_probe", points, other), grain, same) > grain.size() * 9 / 10);
}

namespace {

// a red, b blue: the red channel is how far toward a a point's color is.
constexpr textures::SwirlData red_and_blue(std::uint32_t vanes, float twist) {
    return textures::SwirlData{.a = {1.0f, 0.0f, 0.0f},
                               .vanes = vanes,
                               .b = {0.0f, 0.0f, 1.0f},
                               .twist = twist,
                               .seed = 5u,
                               .padding = {0, 0, 0}};
}

}  // namespace

TEST_CASE("a swirl stays between its colors, and is the same at any distance from its axis") {
    const textures::SwirlData data{.a = {0.9f, 0.3f, 0.03f},
                                   .vanes = 3u,
                                   .b = {0.92f, 0.88f, 0.78f},
                                   .twist = 0.6f,
                                   .seed = 1u,
                                   .padding = {0, 0, 0}};
    tests::ProbeRunner gpu;
    std::vector<TexturePoint> points;
    for (std::uint64_t i = 0; i < 8192; ++i) {
        const double phi = 2.0 * std::numbers::pi * uniform(i, 0);
        const double y = 2.0 * uniform(i, 1) - 1.0;
        const double r = 0.05 + 0.95 * uniform(i, 2);
        points.push_back(at(r * std::cos(phi), y, r * std::sin(phi)));
        points.push_back(at(0.5 * r * std::cos(phi), y, 0.5 * r * std::sin(phi)));
    }
    const std::vector<contracts::Float3> swirled = colors(gpu, "swirl_probe", points, data);
    std::size_t outside = 0;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < swirled.size(); i += 2) {
        for (int c = 0; c < 3; ++c) {
            const float v = contracts::component(swirled[i], c);
            const float a = contracts::component(data.a, c);
            const float b = contracts::component(data.b, c);
            outside += v >= std::min(a, b) - 1e-6f && v <= std::max(a, b) + 1e-6f ? 0u : 1u;
            differ += std::abs(v - contracts::component(swirled[i + 1], c)) > 1e-5f ? 1u : 0u;
        }
    }
    CHECK(outside == 0);
    CHECK(differ == 0);
}

TEST_CASE("a swirl's bands turn twist times round per unit of height, whatever the vanes") {
    // Each band's middle at phi = 2 pi (k / vanes - twist y): a point
    // turned back by 2 pi twist dy and raised dy sits in the same band,
    // edges wavering aside; turned the other way, it does not.
    constexpr std::uint64_t count = 8192;
    tests::ProbeRunner gpu;
    for (const std::uint32_t vanes : {1u, 4u}) {
        const float twist = 0.4f;
        const double dy = 0.2;
        std::vector<TexturePoint> points;
        points.reserve(3 * count);
        for (std::uint64_t i = 0; i < count; ++i) {
            const double phi = 2.0 * std::numbers::pi * uniform(i, 3);
            const double y = -0.5 + uniform(i, 4) * 0.6;
            const double along = phi - 2.0 * std::numbers::pi * twist * dy;
            const double against = phi + 2.0 * std::numbers::pi * twist * dy;
            points.push_back(at(std::cos(phi), y, std::sin(phi)));
            points.push_back(at(std::cos(along), y + dy, std::sin(along)));
            points.push_back(at(std::cos(against), y + dy, std::sin(against)));
        }
        const std::vector<contracts::Float3> bands = colors(gpu, "swirl_probe", points, red_and_blue(vanes, twist));
        double same_band = 0.0;
        double other_way = 0.0;
        for (std::size_t i = 0; i < bands.size(); i += 3) {
            same_band += std::abs(bands[i].x - bands[i + 1].x);
            other_way += std::abs(bands[i].x - bands[i + 2].x);
        }
        same_band /= static_cast<double>(count);
        other_way /= static_cast<double>(count);
        INFO(vanes << " vanes: mean difference following the twist " << same_band << ", against it " << other_way);
        CHECK(same_band < 0.2);
        CHECK(other_way > 2.0 * same_band);
    }
}
