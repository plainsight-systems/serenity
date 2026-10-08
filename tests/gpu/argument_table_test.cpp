// Settles, on this machine, what Apple's documentation leaves unsaid
// (metal/frame/renderer.h): when one Metal 4 argument table is rebound between
// two dispatches in one encoder, each dispatch sees the binding it was
// encoded with. If the GPU read the table when the work ran, both dispatches
// would write the second buffer and the first would stay untouched.

#include <cstdint>
#include <cstring>

#include <doctest/doctest.h>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;

TEST_CASE("rebinding an argument table between dispatches gives each dispatch its own binding") {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::smoke);
    auto pipeline = library.compute_pipeline("smoke");

    constexpr std::uint32_t count = 256;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    auto first = NS::TransferPtr(mtl->newBuffer(count * 4, MTL::ResourceStorageModeShared));
    auto second = NS::TransferPtr(mtl->newBuffer(count * 4, MTL::ResourceStorageModeShared));
    auto counts = NS::TransferPtr(mtl->newBuffer(4, MTL::ResourceStorageModeShared));
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(counts);
    std::memset(first->contents(), 0, count * 4);
    std::memset(second->contents(), 0, count * 4);
    std::memcpy(counts->contents(), &count, 4);
    for (MTL::Buffer* buffer : {first.get(), second.get(), counts.get()}) {
        submission.make_resident(buffer);
    }

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(2);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);

    const auto frame = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    table->setAddress(counts->gpuAddress(), 1);
    table->setAddress(first->gpuAddress(), 0);
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    table->setAddress(second->gpuAddress(), 0);
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    submission.wait_until_complete(frame.sequence);

    const auto* a = static_cast<const std::uint32_t*>(first->contents());
    const auto* b = static_cast<const std::uint32_t*>(second->contents());
    std::uint32_t wrong_first = 0;
    std::uint32_t wrong_second = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        wrong_first += a[i] != 3 * i + 1;
        wrong_second += b[i] != 3 * i + 1;
    }
    CHECK(wrong_first == 0);
    CHECK(wrong_second == 0);
}
