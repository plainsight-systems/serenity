// Film's accumulation (metal/film/accumulate.metal.h), run on the GPU
// (tests/gpu/kernels/film_probe.metal): the running mean and count a pixel
// keeps, and what a sample that is not finite does: left out of its pixel,
// counted once, the pixel's own count unchanged, so the samples it does hold
// are weighed right.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <doctest/doctest.h>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

namespace {

struct Fold {
    std::vector<std::array<float, 4>> pixels;
    std::uint32_t counted = 0;
};

// Folds samples[i] into pixels[i] (starting over where starts[i]) on the
// GPU, and returns the pixels and what the counter counted.
Fold fold(const std::vector<std::array<float, 4>>& pixels, const std::vector<std::array<float, 4>>& samples,
          const std::vector<std::uint32_t>& starts) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline("film_fold");
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const std::uint32_t count = std::uint32_t(pixels.size());

    const auto buffer = [&](const void* data, std::size_t bytes) {
        auto b = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(bytes, 16), MTL::ResourceStorageModeShared));
        REQUIRE(b);
        if (data != nullptr) {
            std::memcpy(b->contents(), data, bytes);
        } else {
            std::memset(b->contents(), 0, bytes);
        }
        submission.make_resident(b.get());
        return b;
    };
    auto out = buffer(nullptr, count * 16);
    auto in = buffer(pixels.data(), count * 16);
    auto sample = buffer(samples.data(), count * 16);
    auto start = buffer(starts.data(), count * 4);
    auto n = buffer(&count, 4);
    auto counter = buffer(nullptr, 4);

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(6);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    MTL::Buffer* bound[] = {out.get(), in.get(), sample.get(), start.get(), n.get(), counter.get()};
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

    Fold result;
    result.pixels.resize(count);
    std::memcpy(result.pixels.data(), out->contents(), count * 16);
    std::memcpy(&result.counted, counter->contents(), 4);
    return result;
}

constexpr float not_a_number = std::numeric_limits<float>::quiet_NaN();
constexpr float infinite = std::numeric_limits<float>::infinity();

}  // namespace

TEST_CASE("a pixel folds a sample into its running mean and counts it") {
    const auto r = fold({{0.5f, 1.0f, 2.0f, 3.0f}, {9.0f, 9.0f, 9.0f, 7.0f}},  // mean 0.5,1,2 of 3; old contents
                        {{1.5f, 1.0f, 0.0f, 0.0f}, {4.0f, 5.0f, 6.0f, 0.0f}}, {0u, 1u});
    // (0.5 x 3 + 1.5) / 4 = 0.75, (1 x 3 + 1) / 4 = 1, (2 x 3 + 0) / 4 = 1.5.
    CHECK(r.pixels[0][0] == doctest::Approx(0.75f));
    CHECK(r.pixels[0][1] == doctest::Approx(1.0f));
    CHECK(r.pixels[0][2] == doctest::Approx(1.5f));
    CHECK(r.pixels[0][3] == 4.0f);
    // Starting over: the old contents are not a mean of anything.
    CHECK(r.pixels[1][0] == 4.0f);
    CHECK(r.pixels[1][2] == 6.0f);
    CHECK(r.pixels[1][3] == 1.0f);
    CHECK(r.counted == 0);
}

TEST_CASE("a sample not finite is left out of its pixel, counted, and the pixel's own count holds") {
    const auto first = fold({{0.0f, 0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 2.0f}},
                            {{not_a_number, 0.0f, 0.0f, 0.0f}, {infinite, 1.0f, 1.0f, 0.0f}, {3.0f, 3.0f, 3.0f, 0.0f}},
                            {1u, 0u, 0u});
    CHECK(first.counted == 2);
    // Starting over and failing: empty, count 0.
    CHECK(first.pixels[0][0] == 0.0f);
    CHECK(first.pixels[0][3] == 0.0f);
    // Failing: the mean and its count as they were.
    CHECK(first.pixels[1][0] == 2.0f);
    CHECK(first.pixels[1][3] == 1.0f);
    // The good one beside them folds in as usual.
    CHECK(first.pixels[2][0] == doctest::Approx(5.0f / 3.0f));
    CHECK(first.pixels[2][3] == 3.0f);

    // Frame 0 not finite, frame 1 radiance L: the pixel shows L, not L / 2.
    const auto second = fold({first.pixels[0]}, {{0.8f, 0.8f, 0.8f, 0.0f}}, {0u});
    CHECK(second.pixels[0][0] == doctest::Approx(0.8f));
    CHECK(second.pixels[0][3] == 1.0f);
}

TEST_CASE("every pixel failing is counted exactly, a SIMD group at a time") {
    constexpr std::uint32_t count = 10000;  // not a multiple of any SIMD width
    const std::vector<std::array<float, 4>> pixels(count, {0.0f, 0.0f, 0.0f, 0.0f});
    const std::vector<std::array<float, 4>> samples(count, {not_a_number, not_a_number, not_a_number, 0.0f});
    const std::vector<std::uint32_t> starts(count, 1u);
    CHECK(fold(pixels, samples, starts).counted == count);
}
