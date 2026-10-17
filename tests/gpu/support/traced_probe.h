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
// encoder as the trace when the shapes are moved, as a frame does. The
// probes' library is the caller's, made once for all its dispatches (P.9,
// GPU.6), as the probe runner's is.
//
// Failures are the test's (P.7), as the probe runner's are.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/transform.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/shared_resources.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"

namespace serenity::tests {

// A scene on the GPU, as a frame traces it: its arrays, its shapes'
// transforms and its structure; and the probes' library (metallib::probes).
struct Traced {
    metal::Device& device;
    metal::Submission& submission;
    const metal::Library& library;
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
    const auto pipeline = traced.library.compute_pipeline(kernel);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = traced.device.handle();
    const auto out = shared_buffer(traced.device, traced.submission, {}, threads * sizeof(Out));
    std::vector<NS::SharedPtr<MTL::Buffer>> buffers;
    for (const Binding& input : inputs) {
        buffers.push_back(shared_buffer(traced.device, traced.submission, input.bytes()));
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
