#include "metal/device/static_arrays.h"

#include <cstring>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// Each array starts on this boundary: more than any shared layout's
// alignment, and what Metal asks of a constant-buffer offset.
constexpr std::size_t alignment = 256;

std::size_t aligned(std::size_t offset) {
    return (offset + alignment - 1) / alignment * alignment;
}

}  // namespace

StaticArrays::StaticArrays(const Device& device, Submission& submission,
                           std::span<const std::span<const std::byte>> arrays) {
    std::vector<std::size_t> offsets;
    offsets.reserve(arrays.size());
    // The first block is the empty arrays' (static_arrays.h).
    std::size_t total = alignment;
    for (std::span<const std::byte> array : arrays) {
        total = aligned(total);
        offsets.push_back(total);
        total += array.size();
    }

    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    buffer_ = NS::TransferPtr(device.handle()->newBuffer(total, MTL::ResourceStorageModeShared));
    if (!buffer_) {
        throw Error("the device made no buffer of " + std::to_string(total) + " bytes for static arrays");
    }
    auto* bytes = static_cast<std::byte*>(buffer_->contents());
    std::memset(bytes, 0, alignment);
    addresses_.assign(arrays.size(), buffer_->gpuAddress());
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        if (arrays[i].empty()) {
            continue;
        }
        std::memcpy(bytes + offsets[i], arrays[i].data(), arrays[i].size());
        addresses_[i] = buffer_->gpuAddress() + offsets[i];
    }
    submission.make_resident(buffer_.get());
}

}  // namespace serenity::metal
