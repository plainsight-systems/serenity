// Shows on this machine what Metal 4's header states and the renderer relies
// on (metal/frame/renderer.h): the encoder "takes a snapshot of the
// resources in the argument table when you make dispatch or execute calls"
// (MTL4ComputeCommandEncoder.h, setArgumentTable), so when one argument
// table is rebound between two dispatches in one encoder, each dispatch sees
// the binding it was encoded with. If the GPU read the table when the work
// ran, both dispatches would write the second buffer and the first would
// stay untouched.
//
// Its own dispatch, not the probe runner's (support/probe_runner.h): the
// rebinding between two dispatches of one encoder is what it tests.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

#include "gpu/support/probe_runner.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "serenity/metallib/probes.h"

using namespace serenity;

TEST_CASE("rebinding an argument table between dispatches gives each dispatch its own binding") {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::probes);
    const auto pipeline = library.compute_pipeline("smoke");

    constexpr std::uint32_t count = 256;
    constexpr std::size_t bytes = count * sizeof(std::uint32_t);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    auto first = NS::TransferPtr(mtl->newBuffer(bytes, MTL::ResourceStorageModeShared));
    REQUIRE(first);
    auto second = NS::TransferPtr(mtl->newBuffer(bytes, MTL::ResourceStorageModeShared));
    REQUIRE(second);
    auto counts = NS::TransferPtr(mtl->newBuffer(sizeof count, MTL::ResourceStorageModeShared));
    REQUIRE(counts);
    std::memset(first->contents(), 0, bytes);
    std::memset(second->contents(), 0, bytes);
    std::memcpy(counts->contents(), &count, sizeof count);
    for (MTL::Buffer* buffer : {first.get(), second.get(), counts.get()}) {
        submission.make_resident(buffer);
    }

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(2);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    INFO("argument table: " << tests::reason(error));
    REQUIRE(table);

    // smoke's bindings (kernels/smoke.metal): out at 0, count at 1.
    const metal::FrameSlot slot = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    table->setAddress(counts->gpuAddress(), 1);
    table->setAddress(first->gpuAddress(), 0);
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    table->setAddress(second->gpuAddress(), 0);
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(slot.sequence);

    const auto written = [&](MTL::Buffer* buffer) {
        std::vector<std::uint32_t> values(count);
        std::memcpy(values.data(), buffer->contents(), bytes);
        std::uint32_t wrong = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            wrong += values[i] != 3 * i + 1 ? 1u : 0u;
        }
        return wrong;
    };
    CHECK(written(first.get()) == 0);
    CHECK(written(second.get()) == 0);
}
