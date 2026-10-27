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
      first_done_(make_event(device_.get())),
      hand_off_ready_(make_event(device_.get())),
      hand_off_done_(make_event(device_.get())) {
  const auto pool = scoped_pool();
  queue_->addResidencySet(residency_.get());
  overlap_->addResidencySet(residency_.get());
  for (Slot& slot : slots_) {
    for (Part* part : {&slot.first, &slot.second, &slot.after}) {
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
  const std::uint64_t committed = frames_.committed();
  return committed == 0 ||
         completed_->waitUntilSignaledValue(committed, timeout.count());
}

std::optional<Completed> Submission::settle(std::uint64_t sequence) {
  Slot& slot = slots_.at(sequence % frames_in_flight);
  if (frames_.is_settled(sequence)) {
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
  // A two-part submission's record spans both parts, a hand-off's to the
  // part after it (submission.h).
  const Feedback* last = &first;
  if (slot.parts == Parts::two) {
    last = slot.second.feedback.get();
    arrived(*last);
  }
  if (slot.handed_off) {
    const HandOffReport& report = *slot.report;
    while (report.arrived_.load(std::memory_order_acquire) < sequence + 1) {
      if (std::chrono::steady_clock::now() > deadline) {
        throw MetalError("no report from submission " +
                         std::to_string(sequence) + "'s hand-off within " +
                         std::to_string(timeout.count()) + " ms");
      }
      std::this_thread::sleep_for(feedback_poll);
    }
    if (report.failed_.load(std::memory_order_acquire)) {
      throw MetalError("submission " + std::to_string(sequence) +
                       "'s hand-off failed: " + report.failure_);
    }
    last = slot.after.feedback.get();
    arrived(*last);
  }
  frames_.settled(sequence);
  return Completed{sequence, first.gpu_start, last->gpu_end};
}

FrameSlot Submission::begin(Parts parts) {
  const FrameSequence::Begun begun = frames_.to_begin();
  const std::uint64_t sequence = begun.sequence;
  const std::uint32_t index = begun.slot;
  Slot& slot = slots_.at(index);

  // The slot is free once the submission that last used it has completed
  // and reported (submission.h).
  std::optional<Completed> settled;
  if (begun.settle_first) {
    settled = settle(*begun.settle_first);
  }

  slot.parts = parts;
  slot.handed_off = false;
  slot.first.allocator->reset();
  slot.first.commands->beginCommandBuffer(slot.first.allocator.get());
  MTL4::CommandBuffer* second = nullptr;
  if (parts == Parts::two) {
    slot.second.allocator->reset();
    slot.second.commands->beginCommandBuffer(slot.second.allocator.get());
    second = slot.second.commands.get();
  }
  frames_.opened();
  return FrameSlot{slot.first.commands.get(), second, index, sequence, settled};
}

HandOff Submission::hand_off() {
  const std::uint64_t sequence = frames_.open();
  Slot& slot = slots_.at(sequence % frames_in_flight);
  if (slot.handed_off) {
    throw MetalError("hand_off(): submission " + std::to_string(sequence) +
                     " has been handed off already");
  }
  slot.handed_off = true;
  slot.report->reset();
  slot.after.allocator->reset();
  slot.after.commands->beginCommandBuffer(slot.after.allocator.get());
  return HandOff{slot.after.commands.get(), hand_off_ready_.get(),
                 hand_off_done_.get(), sequence + 1, slot.report};
}

void HandOffReport::reset() noexcept {
  failed_.store(false, std::memory_order_relaxed);
  failure_.clear();
}

void HandOffReport::succeeded(std::uint64_t sequence) noexcept {
  arrived_.store(sequence + 1, std::memory_order_release);
}

void HandOffReport::failed(std::uint64_t sequence, std::string why) noexcept {
  if (!failed_.load(std::memory_order_relaxed)) {
    failure_ = std::move(why);
    failed_.store(true, std::memory_order_release);
  }
  arrived_.store(sequence + 1, std::memory_order_release);
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
  const std::uint64_t sequence = frames_.commit();
  Slot& slot = slots_.at(sequence % frames_in_flight);
  const bool two = slot.parts == Parts::two;
  const bool handed_off = slot.handed_off;
  slot.first.commands->endCommandBuffer();
  if (two) {
    slot.second.commands->endCommandBuffer();
  }
  if (handed_off) {
    slot.after.commands->endCommandBuffer();
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
  }
  // The last part, split around a hand-off's worker if it has one
  // (submission.h, "A hand-off"); the drawable waited for before the part
  // that writes it, the last.
  Part& last_part = two ? slot.second : slot.first;
  const char* last_name = two ? ", part 2," : "";
  if (handed_off) {
    commit_part(
        last, last_part.commands.get(), last_part.feedback, sequence,
        two ? ", part 2 before its hand-off," : ", before its hand-off,");
    last->signalEvent(hand_off_ready_.get(), sequence + 1);
    last->wait(hand_off_done_.get(), sequence + 1);
  }
  if (drawable != nullptr) {
    last->wait(drawable);
  }
  if (handed_off) {
    commit_part(last, slot.after.commands.get(), slot.after.feedback, sequence,
                ", after its hand-off,");
  } else {
    commit_part(last, last_part.commands.get(), last_part.feedback, sequence,
                last_name);
  }
  if (drawable != nullptr) {
    last->signalDrawable(drawable);
  }
  last->signalEvent(completed_.get(), sequence + 1);
  last_committed_ = slot.parts;
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
  frames_.check_waitable(sequence);
  return settle(sequence);
}

std::vector<Completed> Submission::finish() {
  std::vector<Completed> settled = drain();
  frames_.finish();
  return settled;
}

std::vector<Completed> Submission::drain() {
  // The submissions that may be unsettled: the last frames_in_flight
  // committed (FrameSequence::drain_from).
  const std::uint64_t first = frames_.drain_from();
  std::vector<Completed> settled;
  for (std::uint64_t sequence = first; sequence < frames_.next(); ++sequence) {
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
  frames_.released();  // an open one may have recorded a use of `allocation`
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
  frames_
      .released();  // an open one may have recorded a use of what `set` holds
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
