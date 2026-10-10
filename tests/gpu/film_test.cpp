// Film's accumulation (metal/film/accumulate.metal.h), run on the GPU
// (tests/gpu/kernels/film_probe.metal): the running mean and count a pixel
// keeps, and what a sample that is not finite does: left out of its pixel,
// counted once, the pixel's own count unchanged, so the samples it does hold
// are weighed right.

#include <cstdint>
#include <limits>
#include <vector>

#include <doctest/doctest.h>

#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"

using namespace serenity;
using tests::Binding;
using tests::Float4;

namespace {

// What fold() returns: the pixels, each a mean and its count in w, and what
// the counter counted.
struct Fold {
    std::vector<Float4> pixels;
    std::uint32_t counted = 0;
};

// What each pixel is to do with its sample.
struct Folding {
    std::vector<Float4> pixels;
    std::vector<Float4> samples;
    std::vector<std::uint32_t> starts;  // 1 where the pixel starts over
};

// Folds each sample into its pixel on the GPU.
Fold fold(const Folding& f) {
    REQUIRE(f.samples.size() == f.pixels.size());
    REQUIRE(f.starts.size() == f.pixels.size());
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(f.pixels.size());
    const tests::ProbeRunner::Buffers out = gpu.dispatch(
        "film_fold", count,
        {Binding::zeroed(count * sizeof(Float4)), Binding::of(f.pixels), Binding::of(f.samples),
         Binding::of(f.starts), Binding::of(count), Binding::zeroed(sizeof(std::uint32_t))});
    return {out.read<Float4>(0), out.read<std::uint32_t>(5).at(0)};
}

constexpr float not_a_number = std::numeric_limits<float>::quiet_NaN();
constexpr float infinite = std::numeric_limits<float>::infinity();

}  // namespace

TEST_CASE("a pixel folds a sample into its running mean and counts it") {
    // Mean (0.5, 1, 2) of 3; and old contents.
    const Fold r = fold({.pixels = {{0.5f, 1.0f, 2.0f, 3.0f}, {9.0f, 9.0f, 9.0f, 7.0f}},
                         .samples = {{1.5f, 1.0f, 0.0f, 0.0f}, {4.0f, 5.0f, 6.0f, 0.0f}},
                         .starts = {0u, 1u}});
    // (0.5 x 3 + 1.5) / 4 = 0.75, (1 x 3 + 1) / 4 = 1, (2 x 3 + 0) / 4 = 1.5.
    CHECK(r.pixels[0].x == doctest::Approx(0.75f).scale(0).epsilon(1e-6));
    CHECK(r.pixels[0].y == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(r.pixels[0].z == doctest::Approx(1.5f).scale(0).epsilon(1e-6));
    CHECK(r.pixels[0].w == 4.0f);
    // Starting over: the old contents are not a mean of anything.
    CHECK(r.pixels[1].x == 4.0f);
    CHECK(r.pixels[1].y == 5.0f);
    CHECK(r.pixels[1].z == 6.0f);
    CHECK(r.pixels[1].w == 1.0f);
    CHECK(r.counted == 0);
}

TEST_CASE("a sample not finite is left out of its pixel, counted, and the pixel's own count holds") {
    const Fold first = fold({.pixels = {{0.0f, 0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 2.0f}},
                             .samples = {{not_a_number, 0.0f, 0.0f, 0.0f},
                                         {infinite, 1.0f, 1.0f, 0.0f},
                                         {3.0f, 3.0f, 3.0f, 0.0f}},
                             .starts = {1u, 0u, 0u}});
    CHECK(first.counted == 2);
    // Starting over and failing: empty, count 0.
    CHECK(first.pixels[0].x == 0.0f);
    CHECK(first.pixels[0].w == 0.0f);
    // Failing: the mean and its count as they were.
    CHECK(first.pixels[1].x == 2.0f);
    CHECK(first.pixels[1].w == 1.0f);
    // The good one beside them folds in as usual.
    CHECK(first.pixels[2].x == doctest::Approx(5.0f / 3.0f).scale(0).epsilon(1e-6));
    CHECK(first.pixels[2].w == 3.0f);

    // Frame 0 not finite, frame 1 radiance L: the pixel shows L, not L / 2.
    const Fold second =
        fold({.pixels = {first.pixels[0]}, .samples = {{0.8f, 0.8f, 0.8f, 0.0f}}, .starts = {0u}});
    CHECK(second.pixels[0].x == doctest::Approx(0.8f).scale(0).epsilon(1e-6));
    CHECK(second.pixels[0].w == 1.0f);
}

TEST_CASE("every pixel failing is counted exactly, a SIMD group at a time") {
    constexpr std::uint32_t count = 10000;  // not a multiple of any SIMD width
    const Fold all = fold({.pixels = std::vector<Float4>(count, {0.0f, 0.0f, 0.0f, 0.0f}),
                           .samples = std::vector<Float4>(count, {not_a_number, not_a_number, not_a_number, 0.0f}),
                           .starts = std::vector<std::uint32_t>(count, 1u)});
    CHECK(all.counted == count);
}
