// The procedural textures on the GPU (tests/gpu/kernels/texture_probe.metal),
// held to what their headers promise: the noise (core/textures/noise.h) 0 at
// every lattice point, within its bound, continuous, centered on 0, and a
// function of the point and the seed; the wood (core/textures/wood.h) within
// its bounds and finite across the world, the same at every height, dark at
// the seams, and different from board to board and seed to seed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/textures/noise.h"
#include "core/textures/wood.h"
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

float seed_bits(std::uint32_t seed) {
    float f;
    std::memcpy(&f, &seed, 4);
    return f;
}

double uniform(std::uint64_t i, std::uint64_t which) {
    return animation::draw(0x5EEDu, i, which);
}

std::vector<float> noise(const std::vector<Probe>& points) {
    return run("noise_probe", points, 1);
}

constexpr textures::WoodData walnut{{0.13f, 0.065f, 0.03f}, 0.004f, {0.045f, 0.02f, 0.008f}, 0.16f, 3u, {0, 0, 0}};

std::vector<std::array<float, 3>> wood(const std::vector<Probe>& points, const textures::WoodData& data = walnut) {
    const std::vector<float> raw = run("wood_probe", points, 4, &data, sizeof(data));
    std::vector<std::array<float, 3>> colors(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        colors[i] = {raw[4 * i], raw[4 * i + 1], raw[4 * i + 2]};
    }
    return colors;
}

}  // namespace

TEST_CASE("noise is 0 at every lattice point, whatever the seed") {
    std::vector<Probe> points;
    for (std::uint64_t i = 0; i < 4096; ++i) {
        const auto lattice = [&](std::uint64_t which) { return std::floor(-1000.0 + 2000.0 * uniform(i, which)); };
        points.push_back({float(lattice(0)), float(lattice(1)), float(lattice(2)), seed_bits(std::uint32_t(i))});
    }
    for (float v : noise(points)) {
        CHECK(v == 0.0f);
    }
}

TEST_CASE("noise stays within its bound, centered on 0, and uses its range") {
    std::vector<Probe> points;
    for (std::uint64_t i = 0; i < 1 << 20; ++i) {
        points.push_back({float(-50.0 + 100.0 * uniform(i, 0)), float(-50.0 + 100.0 * uniform(i, 1)),
                          float(-50.0 + 100.0 * uniform(i, 2)), seed_bits(7u)});
    }
    const std::vector<float> values = noise(points);
    double sum = 0.0, largest = 0.0;
    for (float v : values) {
        REQUIRE(std::isfinite(v));
        sum += v;
        largest = std::max(largest, double(std::abs(v)));
    }
    INFO("largest " << largest << ", mean " << sum / values.size());
    CHECK(largest <= textures::noise_bound);
    CHECK(largest > 0.6);
    CHECK(std::abs(sum / values.size()) < 0.01);
}

TEST_CASE("noise is continuous across cells, and a function of the point and the seed") {
    // Pairs a micron apart, many straddling a cell's face.
    std::vector<Probe> points;
    for (std::uint64_t i = 0; i < 65536; ++i) {
        const float x = float(std::floor(-20.0 + 40.0 * uniform(i, 0)) + (i % 2 == 0 ? 0.0 : uniform(i, 3)));
        const float y = float(-20.0 + 40.0 * uniform(i, 1)), z = float(-20.0 + 40.0 * uniform(i, 2));
        points.push_back({x - 1e-4f, y, z, seed_bits(11u)});
        points.push_back({x + 1e-4f, y, z, seed_bits(11u)});
    }
    const std::vector<float> values = noise(points);
    double widest = 0.0;
    for (std::size_t i = 0; i < values.size(); i += 2) {
        widest = std::max(widest, double(std::abs(values[i] - values[i + 1])));
    }
    INFO("widest step over 0.2 mm of noise space: " << widest);
    CHECK(widest < 2e-3);

    // The same points again, the same values; another seed, others.
    CHECK(noise(points) == values);
    std::vector<Probe> reseeded = points;
    for (Probe& p : reseeded) {
        p[3] = seed_bits(12u);
    }
    const std::vector<float> other = noise(reseeded);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        differ += other[i] != values[i];
    }
    CHECK(differ > values.size() * 9 / 10);
}

TEST_CASE("wood stays within its bounds and finite, everywhere in the world") {
    const float lo = textures::wood_seam_shade * (1.0f - textures::wood_pores) * (1.0f - textures::wood_board_shade);
    const float hi = 1.0f + textures::wood_board_shade;
    std::vector<Probe> points;
    for (std::uint64_t i = 0; i < 1 << 18; ++i) {
        // Over the table, and anywhere within a million meters.
        const double reach = i % 2 == 0 ? 2.0 : 1.0e6;
        points.push_back({float(reach * (2.0 * uniform(i, 0) - 1.0)), float(reach * (2.0 * uniform(i, 1) - 1.0)),
                          float(reach * (2.0 * uniform(i, 2) - 1.0)), 0.0f});
    }
    const auto colors = wood(points);
    const float light[3] = {walnut.light.x, walnut.light.y, walnut.light.z};
    const float dark[3] = {walnut.dark.x, walnut.dark.y, walnut.dark.z};
    for (const auto& c : colors) {
        for (int k = 0; k < 3; ++k) {
            REQUIRE(std::isfinite(c[k]));
            CHECK(c[k] >= lo * std::min(light[k], dark[k]) * (1.0f - 1e-5f));
            CHECK(c[k] <= hi * std::max(light[k], dark[k]) * (1.0f + 1e-5f));
        }
    }
}

TEST_CASE("wood is the same at every height, dark at the seams, and differs board to board and seed to seed") {
    std::vector<Probe> points;
    for (std::uint64_t i = 0; i < 4096; ++i) {
        const float x = float(-1.4 + 2.8 * uniform(i, 0)), z = float(-0.9 + 2.6 * uniform(i, 1));
        points.push_back({x, 0.75f, z, 0.0f});
        points.push_back({x, float(-3.0 + 6.0 * uniform(i, 2)), z, 0.0f});
    }
    const auto colors = wood(points);
    for (std::size_t i = 0; i < colors.size(); i += 2) {
        CHECK(colors[i] == colors[i + 1]);
    }

    // Across board 2's seam with board 3, at 0.48 m: a millimetre either
    // side is seam, 3 mm is not.
    std::vector<Probe> seam;
    for (std::uint64_t i = 0; i < 512; ++i) {
        const float x = float(-1.4 + 2.8 * uniform(i, 3));
        for (float dz : {-0.003f, -0.001f, 0.001f, 0.003f}) {
            seam.push_back({x, 0.75f, 0.48f + dz, 0.0f});
        }
    }
    const auto across = wood(seam);
    double in_seam = 0.0, beside = 0.0;
    for (std::size_t i = 0; i < across.size(); i += 4) {
        in_seam += across[i + 1][0] + across[i + 2][0];
        beside += across[i][0] + across[i + 3][0];
    }
    CHECK(in_seam < 0.4 * beside);

    // Each board's mean differs from its neighbor's: each from its own log.
    std::vector<Probe> boards;
    for (int b = 0; b < 8; ++b) {
        for (std::uint64_t i = 0; i < 2048; ++i) {
            boards.push_back({float(-1.4 + 2.8 * uniform(i, 4)), 0.75f,
                              float(0.16 * (b + 0.05 + 0.9 * uniform(i, 5))), 0.0f});
        }
    }
    const auto planks = wood(boards);
    std::array<double, 8> means{};
    for (int b = 0; b < 8; ++b) {
        for (std::size_t i = 0; i < 2048; ++i) {
            means[b] += planks[b * 2048 + i][0] / 2048.0;
        }
    }
    for (int b = 1; b < 8; ++b) {
        INFO("boards " << b - 1 << " and " << b << ": " << means[b - 1] << ", " << means[b]);
        CHECK(std::abs(means[b] - means[b - 1]) > 1e-4);
    }

    textures::WoodData other = walnut;
    other.seed = 4u;
    const auto reseeded = wood(points, other);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < colors.size(); ++i) {
        differ += reseeded[i] != colors[i];
    }
    CHECK(differ > colors.size() * 9 / 10);
}
