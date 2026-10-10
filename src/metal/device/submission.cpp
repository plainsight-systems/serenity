#include "metal/device/submission.h"

#include <chrono>
#include <thread>
#include <utility>

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
    // What this queue's work refers to that this object owns (its allocators,
    // command buffers and residency set) must outlive the work, so wait for
    // it, bounded. Every other owner waits in its own destructor (Lifetime,
    // submission.h). A destructor cannot report what it finds; finish() is
    // how a run reports. Late feedback is harmless: its handler holds the
    // state it writes.
    wait_idle();
}

bool Submission::wait_idle() noexcept {
    const std::uint64_t committed = open_ ? next_ - 1 : next_;
    return committed == 0 || completed_->waitUntilSignaledValue(committed, timeout_ms);
}

std::optional<Completed> Submission::settle(std::uint64_t sequence) {
    Slot& slot = slots_[sequence % frames_in_flight];
    if (slot.settled >= sequence + 1) {
        return std::nullopt;
    }
    if (!completed_->waitUntilSignaledValue(sequence + 1, timeout_ms)) {
        throw Error("submission " + std::to_string(sequence) + " did not complete within " +
                    std::to_string(timeout_ms) + " ms: the GPU has stopped");
    }
    // The event says the GPU is done; the feedback, which carries any error,
    // comes separately and usually just after. Wait for it, bounded.
    const Feedback& feedback = *slot.feedback;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (feedback.arrived.load(std::memory_order_acquire) < sequence + 1) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw Error("no feedback for submission " + std::to_string(sequence) + " within " +
                        std::to_string(timeout_ms) + " ms");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (feedback.failed.load(std::memory_order_acquire)) {
        throw Error(feedback.failure);
    }
    slot.settled = sequence + 1;
    return Completed{sequence, frame::Seconds(feedback.gpu_start), frame::Seconds(feedback.gpu_end)};
}

FrameSlot Submission::begin() {
    if (finished_) {
        throw Error("begin() after finish()");
    }
    if (open_) {
        throw Error("begin() while submission " + std::to_string(next_ - 1) + " is still open");
    }
    const std::uint64_t sequence = next_;
    const std::uint32_t index = static_cast<std::uint32_t>(sequence % frames_in_flight);
    Slot& slot = slots_[index];

    // The slot is free once the submission that last used it has completed
    // and reported (submission.h).
    std::optional<Completed> settled;
    if (sequence >= frames_in_flight) {
        settled = settle(sequence - frames_in_flight);
    }

    slot.allocator->reset();
    slot.commands->beginCommandBuffer(slot.allocator.get());
    open_ = true;
    ++next_;
    return FrameSlot{slot.commands.get(), index, sequence, settled};
}

void Submission::end_and_commit(const MTL::Drawable* drawable) {
    if (!open_) {
        throw Error("commit with no submission begun");
    }
    if (abandoned_) {
        throw Error("submission " + std::to_string(next_ - 1) +
                    " cannot be committed: memory it may use was released while it was open");
    }
    const std::uint64_t sequence = next_ - 1;
    Slot& slot = slots_[sequence % frames_in_flight];
    slot.commands->endCommandBuffer();

    auto drained = pool();
    if (drawable != nullptr) {
        queue_->wait(drawable);
    }

    // Metal 4 reports a submission's GPU error here, on its own queue. The
    // handler holds its own reference to what it writes (submission.h). The
    // options object is made per commit: the API's shape. noexcept: the
    // handler runs inside an Objective-C block on Metal's queue, which an
    // exception must not unwind through, so one (a failed allocation while
    // building the message) ends the program at once instead (E.12).
    auto options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
    options->addFeedbackHandler([state = slot.feedback, sequence](MTL4::CommitFeedback* feedback) noexcept {
        if (feedback != nullptr && feedback->error() != nullptr && !state->failed.load(std::memory_order_relaxed)) {
            state->failure = "submission " + std::to_string(sequence) + " failed on the GPU: " +
                             describe(feedback->error());
            state->failed.store(true, std::memory_order_release);
        }
        if (feedback != nullptr) {
            state->gpu_start = feedback->GPUStartTime();
            state->gpu_end = feedback->GPUEndTime();
        }
        state->arrived.store(sequence + 1, std::memory_order_release);
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

std::optional<Completed> Submission::wait_until_complete(std::uint64_t sequence) {
    if (sequence >= next_ || (open_ && sequence == next_ - 1)) {
        throw Error("wait_until_complete(" + std::to_string(sequence) + ") for a submission not committed");
    }
    if (sequence + frames_in_flight < next_) {
        throw Error("wait_until_complete(" + std::to_string(sequence) + "): its slot has been reused");
    }
    return settle(sequence);
}

std::vector<Completed> Submission::finish() {
    std::vector<Completed> settled = drain();
    finished_ = true;
    return settled;
}

std::vector<Completed> Submission::drain() {
    if (open_) {
        throw Error("drain() or finish() while submission " + std::to_string(next_ - 1) + " is still open");
    }
    // Every submission before these was settled when its slot was reused.
    std::vector<Completed> settled;
    for (std::uint64_t sequence = next_ > frames_in_flight ? next_ - frames_in_flight : 0; sequence < next_;
         ++sequence) {
        if (std::optional<Completed> completed = settle(sequence)) {
            settled.push_back(*completed);
        }
    }
    return settled;
}

bool Submission::has_completed(std::uint64_t sequence) const {
    return completed_->signaledValue() >= sequence + 1;
}

void Submission::make_resident(MTL::Allocation* allocation) {
    if (allocation == nullptr) {
        throw Error("make_resident() with no allocation");
    }
    residency_->addAllocation(allocation);
    residency_->commit();
}

Resident Submission::keep_resident(MTL::Allocation* allocation) {
    make_resident(allocation);
    return Resident(*this, allocation);
}

void Submission::retire(MTL::Allocation* allocation) noexcept {
    wait_idle();
    if (open_) {
        abandoned_ = true;  // it may have recorded a use of `allocation`
    }
    residency_->removeAllocation(allocation);
    residency_->commit();
}

void Submission::add_residency_set(MTL::ResidencySet* set) {
    if (set == nullptr) {
        throw Error("add_residency_set() with no set");
    }
    queue_->addResidencySet(set);
}

void Submission::remove_residency_set(MTL::ResidencySet* set) noexcept {
    wait_idle();
    if (open_) {
        abandoned_ = true;  // it may have recorded a use of what `set` holds
    }
    queue_->removeResidencySet(set);
}

Resident::Resident(Submission& submission, MTL::Allocation* allocation)
    : submission_(&submission), allocation_(NS::RetainPtr(allocation)) {}

Resident::Resident(Resident&& other) noexcept
    : submission_(std::exchange(other.submission_, nullptr)), allocation_(std::move(other.allocation_)) {}

Resident& Resident::operator=(Resident&& other) noexcept {
    if (this != &other) {
        reset();
        submission_ = std::exchange(other.submission_, nullptr);
        allocation_ = std::move(other.allocation_);
    }
    return *this;
}

Resident::~Resident() {
    reset();
}

void Resident::reset() noexcept {
    if (submission_ != nullptr && allocation_) {
        submission_->retire(allocation_.get());
    }
    submission_ = nullptr;
    allocation_.reset();
}

}  // namespace serenity::metal
