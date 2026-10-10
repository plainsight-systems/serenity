#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// An array of bytes the CPU rewrites while frames are in flight: one copy
// per frame slot (submission.h), so the CPU writes the next frame's copy
// while the GPU may still read the last frame's (GPU.7, MEM.4); or one copy,
// shared by every frame, for an array that never changes after start-up. It
// knows nothing of what the bytes are; whoever hands them over says
// (metal/scene/shape_transforms.h). StaticArrays is its counterpart for
// arrays written once (static_arrays.h).
//
// One buffer, in shared storage (Apple silicon's unified memory: the CPU
// writes in place, with no staging buffer and no submission), the copies at
// a stride aligned to 256 bytes, made resident through the submission until
// the array is destroyed (submission.h, Lifetime): one allocation and one
// residency entry (GPU.9). Every copy starts as the bytes given.
//
// When the CPU may write a copy: bytes(slot) is slot `slot`'s, to write only
// between Submission::begin() returning the slot and the frame's commit. The
// frame that last used the slot has then completed (begin() waited for it),
// so the GPU reads nothing in it, and Metal makes the CPU's writes to shared
// memory visible to the GPU at commit. With one copy, every slot names it,
// and it is written only before the first frame.
//
// Throws Error if `initial` is empty, `copies` is neither 1 nor
// frames_in_flight, or the device cannot make the buffer; and bytes() and
// address() throw Error for a slot that is not one (with one copy too, so a
// wrong slot is caught however the array is made).
//
// Cost: the array's size times the copies, once; nothing per frame but what
// the writer writes, and nothing allocated (MEM.9).
class FrameArray {
public:
    FrameArray(const Device& device, Submission& submission, std::span<const std::byte> initial,
               std::uint32_t copies);

    FrameArray(const FrameArray&) = delete;
    FrameArray& operator=(const FrameArray&) = delete;
    FrameArray(FrameArray&&) = delete;
    FrameArray& operator=(FrameArray&&) = delete;
    ~FrameArray() = default;

    // Frame slot `slot`'s copy, for the CPU to write; see above.
    std::span<std::byte> bytes(std::uint32_t slot) const;

    // Frame slot `slot`'s copy, as a shader binds it.
    MTL::GPUAddress address(std::uint32_t slot) const;

private:
    // Where slot `slot`'s copy starts; throws Error for a slot that is not
    // one (I.6).
    std::size_t offset(std::uint32_t slot) const;

    NS::SharedPtr<MTL::Buffer> buffer_;
    Resident resident_;       // released after the GPU is done with it (submission.h)
    std::size_t size_ = 0;    // bytes in each copy
    std::size_t stride_ = 0;  // bytes between copies
    std::uint32_t copies_ = 0;
};

}  // namespace serenity::metal
