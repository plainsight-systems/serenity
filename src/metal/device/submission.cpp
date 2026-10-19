#include "metal/device/submission.h"

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// How long a frame may take before the GPU is taken to have stopped. Far
// beyond any frame this renderer means to draw, so reaching it is a failure,
// not a slow frame.
constexpr std::chrono::milliseconds timeout{5000};

// How long settle() sleeps between looks at a slot's feedback, which almost
// always has arrived by the time the event says the GPU is done.
constexpr std::chrono::microseconds feedback_poll{20};

// The residency set's room before it grows: more than the allocations a
// renderer makes resident (a dozen or so).
constexpr NS::UInteger residency_capacity = 16;

NS::SharedPtr<MTL4::CommandQueue> make_queue(MTL::Device* device) {
  auto queue = NS::TransferPtr(device->newMTL4CommandQueue());
  if (!queue) {
    throw MetalError("the device made no Metal 4 command queue");
  }
  return queue;
}

NS::SharedPtr<MTL::ResidencySet> make_residency_set(MTL::Device* device) {
  const auto pool = scoped_pool();  // Metal's error, if any, is autoreleased
  auto descriptor =
      NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
  descriptor->setInitialCapacity(residency_capacity);
  NS::Error* error = nullptr;
  auto set = NS::TransferPtr(device->newResidencySet(descriptor.get(), &error));
  if (!set) {
    throw MetalError("the device made no residency set: " + describe(error));
  }
  return set;
}

NS::SharedPtr<MTL::SharedEvent> make_event(MTL::Device* device) {
  auto event = NS::TransferPtr(device->newSharedEvent());
  if (!event) {
    throw MetalError("the device made no shared event");
  }
  event->setSignaledValue(0);
  return event;
}

}  // namespace

Submission::Submission(const Device& device)
    : device_(NS::RetainPtr(device.handle())),
      queue_(make_queue(device_.get())),
      overlap_(make_queue(device_.get())),
      residency_(make_residency_set(device_.get())),
      completed_(make_event(device_.get())),
      first_done_(make_event(device_.get())) {
  const auto pool = scoped_pool();
  queue_->addResidencySet(residency_.get());
  overlap_->addResidencySet(residency_.get());
  for (Slot& slot : slots_) {
    for (Part* part : {&slot.first, &slot.second}) {
      part->allocator = NS::TransferPtr(device_->newCommandAllocator());
      part->commands = NS::TransferPtr(device_->newCommandBuffer());
      if (!part->allocator || !part->commands) {
        throw MetalError(
            "the device made no command allocator or command buffer");
      }
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
  return committed == 0 ||
         completed_->waitUntilSignaledValue(committed, timeout.count());
}

std::optional<Completed> Submission::settle(std::uint64_t sequence) {
  Slot& slot = slots_.at(sequence % frames_in_flight);
  if (slot.settled_through >= sequence + 1) {
    return std::nullopt;
  }
  if (!completed_->waitUntilSignaledValue(sequence + 1, timeout.count())) {
    throw MetalError(
        "submission " + std::to_string(sequence) + " did not complete within " +
        std::to_string(timeout.count()) + " ms: the GPU has stopped");
  }
  // The event says the GPU is done; the feedback, which carries any error,
  // comes separately and usually just after, a part's at a time. Wait for
  // each part's, bounded.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  const auto arrived = [&](const Feedback& feedback) {
    while (feedback.arrived.load(std::memory_order_acquire) < sequence + 1) {
      if (std::chrono::steady_clock::now() > deadline) {
        throw MetalError("no feedback for submission " +
                         std::to_string(sequence) + " within " +
                         std::to_string(timeout.count()) + " ms");
      }
      std::this_thread::sleep_for(feedback_poll);
    }
    if (feedback.failed.load(std::memory_order_acquire)) {
      throw MetalError(feedback.failure);
    }
  };
  const Feedback& first = *slot.first.feedback;
  arrived(first);
  // A two-part submission's record spans both parts (submission.h).
  const Feedback& last =
      slot.parts == Parts::two ? *slot.second.feedback : first;
  if (slot.parts == Parts::two) {
    arrived(last);
  }
  slot.settled_through = sequence + 1;
  return Completed{sequence, first.gpu_start, last.gpu_end};
}

FrameSlot Submission::begin(Parts parts) {
  if (finished_) {
    throw MetalError("begin() after finish()");
  }
  if (open_) {
    throw MetalError("begin() while submission " + std::to_string(next_ - 1) +
                     " is still open");
  }
  const std::uint64_t sequence = next_;
  const std::uint32_t index =
      static_cast<std::uint32_t>(sequence % frames_in_flight);
  Slot& slot = slots_.at(index);

  // The slot is free once the submission that last used it has completed
  // and reported (submission.h).
  std::optional<Completed> settled;
  if (sequence >= frames_in_flight) {
    settled = settle(sequence - frames_in_flight);
  }

  slot.parts = parts;
  slot.first.allocator->reset();
  slot.first.commands->beginCommandBuffer(slot.first.allocator.get());
  MTL4::CommandBuffer* second = nullptr;
  if (parts == Parts::two) {
    slot.second.allocator->reset();
    slot.second.commands->beginCommandBuffer(slot.second.allocator.get());
    second = slot.second.commands.get();
  }
  open_ = true;
  ++next_;
  return FrameSlot{slot.first.commands.get(), second, index, sequence, settled};
}

// Commits `commands` to `queue` with a feedback handler writing `state`
// (submission.h, Failure is visible). The handler holds its own reference
// to what it writes. The options object is made per commit: the API's
// shape. noexcept: the handler runs inside an Objective-C block on Metal's
// queue, which an exception must not unwind through, so one (a failed
// allocation while building the message) ends the program at once instead
// (E.12). Called inside an autorelease pool.
void Submission::commit_part(MTL4::CommandQueue* queue,
                             MTL4::CommandBuffer* commands,
                             const std::shared_ptr<Feedback>& state,
                             std::uint64_t sequence, const char* part) {
  auto options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
  options->addFeedbackHandler(
      [state, sequence, part](MTL4::CommitFeedback* feedback) noexcept {
        if (feedback != nullptr && feedback->error() != nullptr &&
            !state->failed.load(std::memory_order_relaxed)) {
          state->failure = "submission " + std::to_string(sequence) + part +
                           " failed on the GPU: " + describe(feedback->error());
          state->failed.store(true, std::memory_order_release);
        }
        if (feedback != nullptr) {
          state->gpu_start = frame::Seconds{feedback->GPUStartTime()};
          state->gpu_end = frame::Seconds{feedback->GPUEndTime()};
        }
        state->arrived.store(sequence + 1, std::memory_order_release);
      });
  const std::array<const MTL4::CommandBuffer*, 1> buffers{commands};
  queue->commit(buffers.data(), buffers.size(), options.get());
}

void Submission::end_and_commit(const MTL::Drawable* drawable) {
  if (!open_) {
    throw MetalError("commit with no submission begun");
  }
  if (abandoned_) {
    throw MetalError("submission " + std::to_string(next_ - 1) +
                     " cannot be committed: memory it may use was released "
                     "while it was open");
  }
  const std::uint64_t sequence = next_ - 1;
  Slot& slot = slots_.at(sequence % frames_in_flight);
  const bool two = slot.parts == Parts::two;
  slot.first.commands->endCommandBuffer();
  if (two) {
    slot.second.commands->endCommandBuffer();
  }

  const auto pool = scoped_pool();
  // The queue of the last part writes the drawable (submission.h).
  MTL4::CommandQueue* const last = two ? overlap_.get() : queue_.get();
  if (!two && last_committed_ == Parts::two) {
    // One part after two: after every part before it (submission.h).
    queue_->wait(completed_.get(), sequence);
  }
  if (two) {
    commit_part(queue_.get(), slot.first.commands.get(), slot.first.feedback,
                sequence, ", part 1,");
    queue_->signalEvent(first_done_.get(), sequence + 1);
    overlap_->wait(first_done_.get(), sequence + 1);
    if (drawable != nullptr) {
      overlap_->wait(drawable);
    }
    commit_part(overlap_.get(), slot.second.commands.get(),
                slot.second.feedback, sequence, ", part 2,");
  } else {
    if (drawable != nullptr) {
      queue_->wait(drawable);
    }
    commit_part(queue_.get(), slot.first.commands.get(), slot.first.feedback,
                sequence, "");
  }
  if (drawable != nullptr) {
    last->signalDrawable(drawable);
  }
  last->signalEvent(completed_.get(), sequence + 1);
  last_committed_ = slot.parts;
  open_ = false;
}

void Submission::commit() { end_and_commit(nullptr); }

void Submission::present(CA::MetalDrawable* drawable) {
  if (drawable == nullptr) {
    throw MetalError("present() with no drawable");
  }
  end_and_commit(drawable);
  drawable->present();
}

std::optional<Completed> Submission::wait_until_complete(
    std::uint64_t sequence) {
  if (sequence >= next_ || (open_ && sequence == next_ - 1)) {
    throw MetalError("wait_until_complete(" + std::to_string(sequence) +
                     ") for a submission not committed");
  }
  if (sequence + frames_in_flight < next_) {
    throw MetalError("wait_until_complete(" + std::to_string(sequence) +
                     "): its slot has been reused");
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
    throw MetalError("drain() or finish() while submission " +
                     std::to_string(next_ - 1) + " is still open");
  }
  // The submissions that may be unsettled: the last frames_in_flight
  // committed. Every one before them was settled when its slot was reused.
  const std::uint64_t first =
      next_ > frames_in_flight ? next_ - frames_in_flight : 0;
  std::vector<Completed> settled;
  for (std::uint64_t sequence = first; sequence < next_; ++sequence) {
    if (std::optional<Completed> completed = settle(sequence)) {
      settled.push_back(*completed);
    }
  }
  return settled;
}

bool Submission::has_completed(std::uint64_t sequence) const noexcept {
  return completed_->signaledValue() >= sequence + 1;
}

void Submission::make_resident(MTL::Allocation* allocation) {
  if (allocation == nullptr) {
    throw MetalError("make_resident() with no allocation");
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
    throw MetalError("add_residency_set() with no set");
  }
  queue_->addResidencySet(set);
  overlap_->addResidencySet(set);
}

void Submission::remove_residency_set(MTL::ResidencySet* set) noexcept {
  wait_idle();
  if (open_) {
    abandoned_ = true;  // it may have recorded a use of what `set` holds
  }
  queue_->removeResidencySet(set);
  overlap_->removeResidencySet(set);
}

Resident::Resident(Submission& submission, MTL::Allocation* allocation)
    : submission_(&submission), allocation_(NS::RetainPtr(allocation)) {}

Resident::Resident(Resident&& other) noexcept
    : submission_(std::exchange(other.submission_, nullptr)),
      allocation_(std::move(other.allocation_)) {}

Resident& Resident::operator=(Resident&& other) noexcept {
  if (this != &other) {
    reset();
    submission_ = std::exchange(other.submission_, nullptr);
    allocation_ = std::move(other.allocation_);
  }
  return *this;
}

Resident::~Resident() { reset(); }

void Resident::reset() noexcept {
  if (submission_ != nullptr && allocation_) {
    submission_->retire(allocation_.get());
  }
  submission_ = nullptr;
  allocation_.reset();
}

}  // namespace serenity::metal
