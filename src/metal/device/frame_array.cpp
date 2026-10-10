#include "metal/device/frame_array.h"

#include <cstring>
#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

std::uint32_t count(FrameArray::Copies copies) {
    // No default: a way of copying with no count fails to compile.
    switch (copies) {
    case FrameArray::Copies::one:
        return 1;
    case FrameArray::Copies::per_frame:
        return frames_in_flight;
    }
    throw MetalError("FrameArray: a number of copies with no count");
}

}  // namespace

FrameArray::FrameArray(const Device& device, Submission& submission, std::span<const std::byte> initial,
                       Copies copies)
    : size_(initial.size()), stride_(align_up(initial.size())), copies_(copies) {
    if (initial.empty()) {
        throw MetalError("FrameArray: an empty array");
    }
    const std::uint32_t total = count(copies);
    const auto pool = scoped_pool();
    buffer_ = NS::TransferPtr(device.handle()->newBuffer(stride_ * total, MTL::ResourceStorageModeShared));
    if (!buffer_) {
        throw MetalError("FrameArray: the device made no buffer of " + std::to_string(stride_ * total) + " bytes");
    }
    auto* bytes = static_cast<std::byte*>(buffer_->contents());
    for (std::uint32_t copy = 0; copy < total; ++copy) {
        std::memcpy(bytes + copy * stride_, initial.data(), size_);
    }
    resident_ = submission.keep_resident(buffer_.get());
}

std::size_t FrameArray::offset(std::uint32_t slot) const {
    if (slot >= frames_in_flight) {
        throw MetalError("FrameArray: no frame slot " + std::to_string(slot));
    }
    return copies_ == Copies::one ? 0 : std::size_t{slot} * stride_;
}

std::span<std::byte> FrameArray::bytes(std::uint32_t slot) {
    return {static_cast<std::byte*>(buffer_->contents()) + offset(slot), size_};
}

MTL::GPUAddress FrameArray::address(std::uint32_t slot) const {
    return buffer_->gpuAddress() + offset(slot);
}

}  // namespace serenity::metal
