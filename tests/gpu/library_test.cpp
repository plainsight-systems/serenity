// metal::Library: a compiled-in library loads, its kernel runs and writes
// what it should, and every failure throws metal::Error naming its cause.
//
// Its own dispatch, through a plain Metal command queue rather than the
// probe runner (support/probe_runner.h): the library is what it tests, so
// it is run with nothing of the backend's around it.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/library.h"
#include "serenity/metallib/probes.h"

using serenity::metal::Device;
using serenity::metal::Error;
using serenity::metal::Library;

TEST_CASE("a compiled-in kernel runs and writes what it should") {
    Device device;
    Library library(device, serenity::metallib::probes);
    const auto pipeline = library.compute_pipeline("smoke");
    REQUIRE(pipeline);

    constexpr std::uint32_t count = 1000;  // not a multiple of the thread width
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto out =
        NS::TransferPtr(device.handle()->newBuffer(count * sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    REQUIRE(out);
    auto queue = NS::TransferPtr(device.handle()->newCommandQueue());
    REQUIRE(queue);

    // smoke's bindings (kernels/smoke.metal): out at 0, count at 1.
    MTL::CommandBuffer* commands = queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(out.get(), 0, 0);
    encoder->setBytes(&count, sizeof(count), 1);
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();
    REQUIRE(commands->status() == MTL::CommandBufferStatusCompleted);

    std::vector<std::uint32_t> values(count);
    std::memcpy(values.data(), out->contents(), count * sizeof(std::uint32_t));
    std::uint32_t wrong = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        wrong += values[i] != 3 * i + 1 ? 1u : 0u;
    }
    CHECK(wrong == 0);
}

TEST_CASE("a function the library lacks is an error that names it") {
    Device device;
    Library library(device, serenity::metallib::probes);
    CHECK_THROWS_WITH_AS((void)library.compute_pipeline("no_such_kernel"), doctest::Contains("no_such_kernel"), Error);
}

TEST_CASE("a null function name is an error") {
    Device device;
    Library library(device, serenity::metallib::probes);
    CHECK_THROWS_AS((void)library.compute_pipeline(nullptr), Error);
}

TEST_CASE("bytes that are not a library are an error") {
    Device device;
    const std::vector<std::byte> garbage(4096, std::byte{0x5a});
    CHECK_THROWS_AS(Library(device, garbage), Error);
    CHECK_THROWS_AS(Library(device, std::span<const std::byte>()), Error);
}
