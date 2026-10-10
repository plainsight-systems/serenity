// The numbers a path draws (metal/sampler/sampler.metal.h), on the GPU
// (tests/gpu/kernels/sampler_probe.metal), held to principle 2
// (logical-overview.md): each a function of pixel, frame and dimension
// alone, so any frame can be drawn again exactly, alone; a stream per pixel
// and frame, none shared; and, over many pixels, uniform on [0, 1), with no
// dimension, frame or neighbouring pixel predicting another.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"

using namespace serenity;
using tests::Binding;
using tests::PathNumbersDraw;
using tests::PathNumbersQuery;
using tests::path_numbers_drawn;

namespace {

// PCG's output permutation, as the shaders compute it (metal/math/
// hash.metal.h): to show which pixels the sampler's old 32-bit key merged.
std::uint32_t pcg_hash(std::uint32_t v) {
    const std::uint32_t state = v * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

std::vector<PathNumbersDraw> draw(tests::ProbeRunner& gpu, const std::vector<PathNumbersQuery>& queries) {
    const auto count = static_cast<std::uint32_t>(queries.size());
    return gpu.run<PathNumbersDraw>("path_numbers_probe", queries.size(),
                                    {Binding::of(queries), Binding::of(count)});
}

bool same(const PathNumbersDraw& a, const PathNumbersDraw& b) {
    for (std::uint32_t k = 0; k < path_numbers_drawn; ++k) {
        if (a.u[k] != b.u[k]) {
            return false;
        }
    }
    return true;
}

// The correlation of a[i] and b[i] over i.
double correlation(const std::vector<double>& a, const std::vector<double>& b) {
    const auto n = static_cast<double>(a.size());
    double sa = 0.0;
    double sb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sa += a[i];
        sb += b[i];
    }
    const double ma = sa / n;
    const double mb = sb / n;
    double cov = 0.0;
    double va = 0.0;
    double vb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma) * (a[i] - ma);
        vb += (b[i] - mb) * (b[i] - mb);
    }
    return cov / std::sqrt(va * vb);
}

}  // namespace

TEST_CASE("a path's numbers: a stream per pixel and frame, none shared, each drawn again exactly") {
    // Pixels (2627, 65) and (0, 818) of frame 0: the old key, a 32-bit hash
    // of pixel and frame, was the same for both, and so were their paths'
    // numbers, all the way down (GDSA.3). The stream is now (pixel, frame)
    // itself (metal/sampler/sampler.metal.h).
    const auto old_key = [](std::uint32_t x, std::uint32_t y, std::uint32_t frame) {
        return pcg_hash(x ^ pcg_hash(y ^ pcg_hash(frame)));
    };
    REQUIRE(old_key(2627, 65, 0) == old_key(0, 818, 0));

    tests::ProbeRunner gpu;
    const std::vector<PathNumbersDraw> numbers =
        draw(gpu, {{2627, 65, 0, 0}, {0, 818, 0, 0}, {2627, 65, 0, 0}, {2627, 65, 1, 0}, {65, 2627, 0, 0}});
    for (const PathNumbersDraw& path : numbers) {
        for (const float u : path.u) {
            CHECK(u >= 0.0f);
            CHECK(u < 1.0f);
        }
    }
    CHECK_FALSE(same(numbers[0], numbers[1]));  // the pixels the old key merged
    CHECK(same(numbers[0], numbers[2]));        // the same pixel and frame, drawn again
    CHECK_FALSE(same(numbers[0], numbers[3]));  // the next frame
    CHECK_FALSE(same(numbers[0], numbers[4]));  // x and y swapped
    // Within a path, no number repeats the one before.
    for (std::uint32_t k = 1; k < path_numbers_drawn; ++k) {
        CHECK(numbers[0].u[k] != numbers[0].u[k - 1]);
    }
}

TEST_CASE("a frame's numbers are drawn again exactly alone, whatever was drawn before them") {
    // Frame 5 of a 64 x 64 image, drawn alone, and drawn after frames 0 to 4
    // in the same dispatch: the same numbers (principle 2).
    constexpr std::uint32_t side = 64;
    std::vector<PathNumbersQuery> alone;
    std::vector<PathNumbersQuery> after;
    for (std::uint32_t frame = 0; frame <= 5; ++frame) {
        for (std::uint32_t y = 0; y < side; ++y) {
            for (std::uint32_t x = 0; x < side; ++x) {
                after.push_back({x, y, frame, 0});
                if (frame == 5) {
                    alone.push_back({x, y, frame, 0});
                }
            }
        }
    }
    tests::ProbeRunner gpu;
    const std::vector<PathNumbersDraw> by_itself = draw(gpu, alone);
    const std::vector<PathNumbersDraw> in_sequence = draw(gpu, after);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < by_itself.size(); ++i) {
        differ += same(by_itself[i], in_sequence[std::size_t{5} * side * side + i]) ? 0u : 1u;
    }
    CHECK(differ == 0);
}

TEST_CASE("over many pixels the numbers are uniform, and no dimension, frame or neighbour predicts another") {
    // 256 x 256 pixels, frames 0 and 1. A uniform number's mean is 1/2 with
    // a standard deviation of 1 / sqrt(12 N); the correlation of two
    // independent ones is 0 with one of 1 / sqrt(N). Each is held to 5 of
    // those: a sampler whose dimensions, frames or pixels shared a stream
    // would be correlated near 1.
    constexpr std::uint32_t side = 256;
    std::vector<PathNumbersQuery> queries;
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
        for (std::uint32_t y = 0; y < side; ++y) {
            for (std::uint32_t x = 0; x < side; ++x) {
                queries.push_back({x, y, frame, 0});
            }
        }
    }
    tests::ProbeRunner gpu;
    const std::vector<PathNumbersDraw> drawn = draw(gpu, queries);
    constexpr std::size_t pixels = std::size_t{side} * side;
    const auto n = static_cast<double>(pixels);
    // dimension k of frame `frame`, over the pixels; its right neighbour's.
    const auto column = [&](std::uint32_t frame, std::uint32_t k, std::uint32_t shift) {
        std::vector<double> values;
        values.reserve(pixels);
        for (std::size_t i = 0; i < pixels; ++i) {
            const std::size_t x = (i % side + shift) % side;
            values.push_back(drawn[frame * pixels + (i / side) * side + x].u[k]);
        }
        return values;
    };
    for (std::uint32_t k = 0; k < path_numbers_drawn; ++k) {
        INFO("dimension " << k);
        const std::vector<double> u = column(0, k, 0);
        double sum = 0.0;
        for (const double v : u) {
            sum += v;
        }
        CHECK(std::abs(sum / n - 0.5) < 5.0 / std::sqrt(12.0 * n));
        CHECK(std::abs(correlation(u, column(1, k, 0))) < 5.0 / std::sqrt(n));  // the next frame
        CHECK(std::abs(correlation(u, column(0, k, 1))) < 5.0 / std::sqrt(n));  // the pixel beside
        if (k + 1 < path_numbers_drawn) {
            CHECK(std::abs(correlation(u, column(0, k + 1, 0))) < 5.0 / std::sqrt(n));  // the next dimension
        }
    }
}
