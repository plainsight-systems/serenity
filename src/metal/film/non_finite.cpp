#include "metal/film/non_finite.h"

#include <atomic>
#include <cstring>

#include "metal/device/error.h"

namespace serenity::metal {

NonFinite::NonFinite(const Device& device, Submission& submission) {
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    counter_ = NS::TransferPtr(device.handle()->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
    if (!counter_) {
        throw Error("the device made no buffer for the count of samples not finite");
    }
    std::memset(counter_->contents(), 0, sizeof(std::uint32_t));
    submission.make_resident(counter_.get());
}

std::uint32_t NonFinite::count() const {
    // Shared memory the GPU writes atomically; read as an atomic, so the
    // value is one the GPU stored (std::atomic_ref, C++20).
    auto* value = static_cast<std::uint32_t*>(counter_->contents());
    return std::atomic_ref<std::uint32_t>(*value).load(std::memory_order_acquire);
}

}  // namespace serenity::metal
