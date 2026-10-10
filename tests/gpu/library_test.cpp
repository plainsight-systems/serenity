// metal::Library: a compiled-in library loads, its kernel runs and writes
// what it should, and every failure throws metal::MetalError naming its cause.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/library.h"
#include "serenity/metallib/smoke.h"

using serenity::metal::Device;
using serenity::metal::MetalError;
using serenity::metal::Library;

TEST_CASE("a compiled-in kernel runs and writes what it should") {
    Device device;
    Library library(device, serenity::metallib::smoke);
    auto pipeline = library.compute_pipeline("smoke");
    REQUIRE(pipeline);

    constexpr std::uint32_t count = 1000;  // not a multiple of the thread width
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto out = NS::TransferPtr(
        device.handle()->newBuffer(count * sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    auto queue = NS::TransferPtr(device.handle()->newCommandQueue());
    REQUIRE(out);
    REQUIRE(queue);

    MTL::CommandBuffer* commands = queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(out.get(), 0, 0);
    encoder->setBytes(&count, sizeof(count), 1);
    encoder->dispatchThreads(MTL::Size(count, 1, 1),
                             MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();
    REQUIRE(commands->status() == MTL::CommandBufferStatusCompleted);

    const auto* values = static_cast<const std::uint32_t*>(out->contents());
    std::uint32_t wrong = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        wrong += values[i] != 3 * i + 1;
    }
    CHECK(wrong == 0);
}

TEST_CASE("a function the library lacks is an error that names it") {
    Device device;
    Library library(device, serenity::metallib::smoke);
    try {
        (void)library.compute_pipeline("no_such_kernel");
        FAIL("expected metal::MetalError");
    } catch (const MetalError& error) {
        CHECK(std::string(error.what()).find("no_such_kernel") != std::string::npos);
    }
}

TEST_CASE("a null function name is an error") {
    Device device;
    Library library(device, serenity::metallib::smoke);
    CHECK_THROWS_AS((void)library.compute_pipeline(nullptr), MetalError);
}

TEST_CASE("bytes that are not a library are an error") {
    Device device;
    const std::vector<std::byte> garbage(4096, std::byte{0x5a});
    CHECK_THROWS_AS(Library(device, garbage), MetalError);
    CHECK_THROWS_AS(Library(device, std::span<const std::byte>()), MetalError);
}
