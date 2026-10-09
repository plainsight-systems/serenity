#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// Arrays of bytes, copied once into GPU memory and made resident, and the
// address of each: how data the CPU writes once and shaders only read
// reaches the GPU. It knows nothing of what the bytes are; whoever hands
// them over says (metal/scene/scene_buffers.h).
//
// One buffer holds every array, each at an offset aligned to 256 bytes, so
// the arrays are one allocation and one residency entry, not one each
// (GPU.9). The buffer is in shared storage, which Apple silicon's unified
// memory allows, so the copy is the CPU's own memcpy, with no staging buffer
// and no submission. An empty array has no bytes of its own; its address is
// that of a zeroed block at the buffer's start, shared by every empty array,
// because Metal refuses a null address for a buffer a shader declares.
//
// Throws Error if the device cannot make the buffer.
//
// Not performance-sensitive: one copy, at start-up.
class StaticArrays {
public:
    // Copies each of `arrays` into a buffer on `device`, made resident
    // through `submission`.
    StaticArrays(const Device& device, Submission& submission, std::span<const std::span<const std::byte>> arrays);

    StaticArrays(const StaticArrays&) = delete;
    StaticArrays& operator=(const StaticArrays&) = delete;
    StaticArrays(StaticArrays&&) = delete;
    StaticArrays& operator=(StaticArrays&&) = delete;
    ~StaticArrays() = default;

    // The address of array `i`, in the order given; the shared zeroed block
    // if it was empty.
    MTL::GPUAddress address(std::size_t i) const { return addresses_.at(i); }

private:
    NS::SharedPtr<MTL::Buffer> buffer_;
    std::vector<MTL::GPUAddress> addresses_;
};

}  // namespace serenity::metal
