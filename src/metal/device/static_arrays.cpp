#include "metal/device/static_arrays.h"

#include <cstring>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

StaticArrays::StaticArrays(const Device& device, Submission& submission,
                           std::span<const std::span<const std::byte>> arrays) {
    std::vector<std::size_t> offsets;
    offsets.reserve(arrays.size());
    // The first block is the empty arrays' (static_arrays.h).
    std::size_t total = buffer_alignment;
    for (std::span<const std::byte> array : arrays) {
        total = align_up(total);
        offsets.push_back(total);
        total += array.size();
    }

    const auto pool = scoped_pool();
    buffer_ = NS::TransferPtr(device.handle()->newBuffer(total, MTL::ResourceStorageModeShared));
    if (!buffer_) {
        throw Error("the device made no buffer of " + std::to_string(total) + " bytes for static arrays");
    }
    auto* bytes = static_cast<std::byte*>(buffer_->contents());
    std::memset(bytes, 0, buffer_alignment);
    addresses_.assign(arrays.size(), buffer_->gpuAddress());
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        if (!arrays[i].empty()) {
            std::memcpy(bytes + offsets[i], arrays[i].data(), arrays[i].size());
            addresses_[i] = buffer_->gpuAddress() + offsets[i];
        }
    }
    resident_ = submission.keep_resident(buffer_.get());
}

MTL::GPUAddress StaticArrays::address(std::size_t i) const {
    if (i >= addresses_.size()) {
        throw Error("StaticArrays: no array " + std::to_string(i) + " of " + std::to_string(addresses_.size()));
    }
    return addresses_[i];
}

}  // namespace serenity::metal
