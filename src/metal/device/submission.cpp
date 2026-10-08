#include "metal/device/submission.h"

#include <chrono>
#include <thread>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// How long a frame may take before the GPU is taken to have stopped. Far
// beyond any frame this renderer means to draw, so reaching it is a failure,
// not a slow frame.
constexpr std::uint64_t timeout_ms = 5000;

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

NS::SharedPtr<NS::AutoreleasePool> pool() {
    return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

}  // namespace

Submission::Submission(const Device& device) : device_(NS::RetainPtr(device.handle())) {
    auto drained = pool();

    queue_ = NS::TransferPtr(device_->newMTL4CommandQueue());
    if (!queue_) {
        throw Error("the device made no Metal 4 command queue");
    }

    auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    descriptor->setInitialCapacity(16);
    NS::Error* error = nullptr;
    residency_ = NS::TransferPtr(device_->newResidencySet(descriptor.get(), &error));
    if (!residency_) {
        throw Error("the device made no residency set: " + describe(error));
    }
    queue_->addResidencySet(residency_.get());

    completed_ = NS::TransferPtr(device_->newSharedEvent());
    if (!completed_) {
        throw Error("the device made no shared event");
    }
    completed_->setSignaledValue(0);

    for (Slot& slot : slots_) {
        slot.allocator = NS::TransferPtr(device_->newCommandAllocator());
        slot.commands = NS::TransferPtr(device_->newCommandBuffer());
        if (!slot.allocator || !slot.commands) {
            throw Error("the device made no command allocator or command buffer");
        }
    }
}

Submission::~Submission() {
    // Everything this queue's work refers to is owned by objects that outlive
    // it only if the work has finished, so wait for it. A destructor cannot
    // report a timeout; the wait is bounded, and whatever it finds is lost.
    const std::uint64_t committed = open_ ? next_ - 1 : next_;
    if (committed > 0) {
        completed_->waitUntilSignaledValue(committed, timeout_ms);
    }
    // The feedback handlers write into the slots, so none may still be due
    // when the slots go. Each slot's last submission is the newest one using
    // it; wait for its feedback, bounded likewise.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (std::uint64_t sequence = next_ > frames_in_flight ? next_ - frames_in_flight : 0; sequence < next_;
         ++sequence) {
        if (open_ && sequence == next_ - 1) {
            break;  // begun, never committed: no handler was registered
        }
        const Slot& slot = slots_[sequence % frames_in_flight];
        while (slot.feedback.load(std::memory_order_acquire) < sequence + 1 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

void Submission::wait_for(std::uint64_t sequence) {
    if (!completed_->waitUntilSignaledValue(sequence + 1, timeout_ms)) {
        throw Error("submission " + std::to_string(sequence) + " did not complete within " +
                    std::to_string(timeout_ms) + " ms: the GPU has stopped");
    }
}

void Submission::check(const Slot& slot) const {
    if (slot.failed.load(std::memory_order_acquire)) {
        throw Error(slot.failure);
    }
}

FrameSlot Submission::begin() {
    if (open_) {
        throw Error("begin() while submission " + std::to_string(next_ - 1) + " is still open");
    }
    const std::uint64_t sequence = next_;
    const std::uint32_t index = static_cast<std::uint32_t>(sequence % frames_in_flight);
    Slot& slot = slots_[index];

    if (sequence >= frames_in_flight) {
        wait_for(sequence - frames_in_flight);
    }
    // Feedback for the slot's last submission may arrive after its event; a
    // failure that arrives later is caught the next time the slot comes round.
    check(slot);

    slot.allocator->reset();
    slot.commands->beginCommandBuffer(slot.allocator.get());
    open_ = true;
    ++next_;
    return FrameSlot{slot.commands.get(), index, sequence};
}

void Submission::end_and_commit(const MTL::Drawable* drawable) {
    if (!open_) {
        throw Error("commit with no submission begun");
    }
    const std::uint64_t sequence = next_ - 1;
    Slot& slot = slots_[sequence % frames_in_flight];
    slot.commands->endCommandBuffer();

    auto drained = pool();
    if (drawable != nullptr) {
        queue_->wait(drawable);
    }

    // Metal 4 reports a submission's GPU error here, on its own queue. The
    // options object is made per commit: the API's shape (submission.h).
    auto options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
    Slot* target = &slot;
    options->addFeedbackHandler([target, sequence](MTL4::CommitFeedback* feedback) {
        if (feedback != nullptr && feedback->error() != nullptr && !target->failed.load(std::memory_order_relaxed)) {
            target->failure = "submission " + std::to_string(sequence) + " failed on the GPU: " +
                              describe(feedback->error());
            target->failed.store(true, std::memory_order_release);
        }
        target->feedback.store(sequence + 1, std::memory_order_release);
    });

    const MTL4::CommandBuffer* buffers[] = {slot.commands.get()};
    queue_->commit(buffers, 1, options.get());
    if (drawable != nullptr) {
        queue_->signalDrawable(drawable);
    }
    queue_->signalEvent(completed_.get(), sequence + 1);
    open_ = false;
}

void Submission::commit() {
    end_and_commit(nullptr);
}

void Submission::present(CA::MetalDrawable* drawable) {
    if (drawable == nullptr) {
        throw Error("present() with no drawable");
    }
    end_and_commit(drawable);
    drawable->present();
}

void Submission::wait_until_complete(std::uint64_t sequence) {
    if (sequence >= next_ || (open_ && sequence == next_ - 1)) {
        throw Error("wait_until_complete(" + std::to_string(sequence) + ") for a submission not committed");
    }
    if (sequence + frames_in_flight < next_) {
        throw Error("wait_until_complete(" + std::to_string(sequence) + "): its slot has been reused");
    }
    wait_for(sequence);

    // The event says the GPU is done; the feedback, which carries any error,
    // comes separately. Wait for it too, bounded, so a readback never reads
    // the image of a submission that failed.
    const Slot& slot = slots_[sequence % frames_in_flight];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (slot.feedback.load(std::memory_order_acquire) < sequence + 1) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw Error("no feedback for submission " + std::to_string(sequence) + " within " +
                        std::to_string(timeout_ms) + " ms");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    check(slot);
}

void Submission::make_resident(MTL::Allocation* allocation) {
    if (allocation == nullptr) {
        throw Error("make_resident() with no allocation");
    }
    residency_->addAllocation(allocation);
    residency_->commit();
}

}  // namespace serenity::metal
