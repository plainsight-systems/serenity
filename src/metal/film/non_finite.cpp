#include "metal/film/non_finite.h"

#include <cstring>

#include "metal/device/error.h"

namespace serenity::metal {

NonFinite::NonFinite(const Device& device, Submission& submission) : submission_(submission) {
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    counters_ = NS::TransferPtr(
        device.handle()->newBuffer(sizeof(std::uint32_t) * frames_in_flight, MTL::ResourceStorageModeShared));
    if (!counters_) {
        throw Error("the device made no buffer for the counts of samples not finite");
    }
    std::memset(counters_->contents(), 0, sizeof(std::uint32_t) * frames_in_flight);
    submission.make_resident(counters_.get());
}

std::uint32_t* NonFinite::counter(std::uint32_t slot) const {
    return static_cast<std::uint32_t*>(counters_->contents()) + slot;
}

void NonFinite::begin_frame(std::uint32_t slot, std::uint64_t sequence) {
    if (slot >= frames_in_flight) {
        throw Error("NonFinite::begin_frame: no slot " + std::to_string(slot));
    }
    // The slot's last frame has completed: begin() waited for it.
    if (counting_[slot] != 0) {
        total_ += *counter(slot);
    }
    *counter(slot) = 0;
    counting_[slot] = sequence + 1;
}

MTL::GPUAddress NonFinite::address(std::uint32_t slot) const {
    return counters_->gpuAddress() + sizeof(std::uint32_t) * slot;
}

std::uint64_t NonFinite::count() const {
    std::uint64_t count = total_;
    for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
        if (counting_[slot] != 0 && submission_.has_completed(counting_[slot] - 1)) {
            count += *counter(slot);
        }
    }
    return count;
}

}  // namespace serenity::metal
