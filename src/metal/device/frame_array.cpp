#include "metal/device/frame_array.h"

#include <cstring>
#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// Each copy starts on this boundary: more than any shared layout's
// alignment, and what Metal asks of a constant-buffer offset.
constexpr std::size_t alignment = 256;

}  // namespace

FrameArray::FrameArray(const Device& device, Submission& submission, std::span<const std::byte> initial,
                       std::uint32_t copies)
    : size_(initial.size()), stride_((initial.size() + alignment - 1) / alignment * alignment), copies_(copies) {
    if (initial.empty()) {
        throw Error("FrameArray: an empty array");
    }
    if (copies != 1 && copies != frames_in_flight) {
        throw Error("FrameArray: " + std::to_string(copies) + " copies; one, or one per frame in flight");
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    buffer_ = NS::TransferPtr(device.handle()->newBuffer(stride_ * copies, MTL::ResourceStorageModeShared));
    if (!buffer_) {
        throw Error("FrameArray: the device made no buffer of " + std::to_string(stride_ * copies) + " bytes");
    }
    auto* bytes = static_cast<std::byte*>(buffer_->contents());
    for (std::uint32_t copy = 0; copy < copies; ++copy) {
        std::memcpy(bytes + copy * stride_, initial.data(), size_);
    }
    resident_ = submission.keep_resident(buffer_.get());
}

std::size_t FrameArray::offset(std::uint32_t slot) const {
    if (slot >= frames_in_flight) {
        throw Error("FrameArray: no frame slot " + std::to_string(slot));
    }
    return copies_ == 1 ? 0 : std::size_t{slot} * stride_;
}

std::span<std::byte> FrameArray::bytes(std::uint32_t slot) const {
    return {static_cast<std::byte*>(buffer_->contents()) + offset(slot), size_};
}

MTL::GPUAddress FrameArray::address(std::uint32_t slot) const {
    return buffer_->gpuAddress() + offset(slot);
}

}  // namespace serenity::metal
