#pragma once

// Runs one probe kernel that traces the scene's acceleration structure, the
// way every such probe test does (ES.3, F.10): the kernel's output at
// buffer 0, its inputs from 1 in the order given, then the structure and the
// shapes' records, transforms and boxes; one dispatch; a wait; and the
// output read back. Each tracing kernel's comment lists its bindings in that
// order (kernels/trace_probe.metal, kernels/leave_probe.metal).
//
// Its own dispatch, not the probe runner's (probe_runner.h): it binds the
// structure as a resource, and records the structure's update in the same
// encoder as the trace when the shapes are moved, as a frame does.
//
// Failures are the test's (P.7), as the probe runner's are.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/transform.h"
#include "gpu/support/probe_runner.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"
#include "serenity/metallib/probes.h"

namespace serenity::tests {

// A scene on the GPU, as a frame traces it: its arrays, its shapes'
// transforms and its structure.
struct Traced {
    metal::Device& device;
    metal::Submission& submission;
    const metal::SceneBuffers& buffers;
    metal::ShapeTransforms& transforms;
    metal::SceneAcceleration& acceleration;
};

// Dispatches `threads` threads of `kernel` over `traced`'s structure for
// the next frame's slot, its output, one Out a thread, at binding 0 and
// `inputs` from 1, in one command buffer; if `moved` is given, first writes
// them as the slot's transforms and records the structure's update.
template <GpuBytes Out>
std::vector<Out> run_traced(const Traced& traced, const char* kernel, std::size_t threads,
                            std::initializer_list<Binding> inputs,
                            const std::optional<std::vector<contracts::Transform>>& moved = std::nullopt) {
    metal::Library library(traced.device, metallib::probes);
    const auto pipeline = library.compute_pipeline(kernel);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = traced.device.handle();
    auto out = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(threads * sizeof(Out), 16),
                                              MTL::ResourceStorageModeShared));
    REQUIRE(out);
    std::memset(out->contents(), 0, threads * sizeof(Out));
    traced.submission.make_resident(out.get());
    std::vector<NS::SharedPtr<MTL::Buffer>> buffers;
    for (const Binding& input : inputs) {
        // At least 16 bytes, so an empty array still has an address.
        auto buffer = NS::TransferPtr(
            mtl->newBuffer(std::max<std::size_t>(input.bytes().size(), 16), MTL::ResourceStorageModeShared));
        REQUIRE(buffer);
        std::memcpy(buffer->contents(), input.bytes().data(), input.bytes().size());
        traced.submission.make_resident(buffer.get());
        buffers.push_back(std::move(buffer));
    }
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(1 + buffers.size() + 4);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    INFO("argument table: " << reason(error));
    REQUIRE(table);

    const metal::FrameSlot slot = traced.submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    if (moved) {
        const std::span<contracts::Transform> placed = traced.transforms.transforms(slot.slot);
        std::ranges::copy(*moved, placed.begin());
        traced.acceleration.update(encoder, slot.slot, placed);
    }
    std::size_t index = 0;
    table->setAddress(out->gpuAddress(), index++);
    for (const auto& buffer : buffers) {
        table->setAddress(buffer->gpuAddress(), index++);
    }
    table->setResource(traced.acceleration.resource(slot.slot), index++);
    table->setAddress(traced.buffers.block().shapes, index++);
    table->setAddress(traced.transforms.address(slot.slot), index++);
    table->setAddress(traced.buffers.block().boxes, index);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    traced.submission.commit();
    (void)traced.submission.wait_until_complete(slot.sequence);

    std::vector<Out> results(threads);
    std::memcpy(results.data(), out->contents(), threads * sizeof(Out));
    return results;
}

}  // namespace serenity::tests
