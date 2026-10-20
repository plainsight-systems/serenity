#include "cuda/device/submission.h"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// Every event of `events` made, or none: on a failure, those made are
// destroyed and the failure thrown (E.6: nothing leaks from a constructor
// that throws).
void make_events(std::span<CUevent> events, unsigned flags) {
  for (CUevent& event : events) {
    if (const CUresult made = cuEventCreate(&event, flags);
        made != CUDA_SUCCESS) {
      for (CUevent& made_one : events) {
        if (made_one != nullptr) {
          (void)cuEventDestroy(made_one);
          made_one = nullptr;
        }
      }
      check(made, "cuEventCreate");
    }
  }
}

void destroy_events(std::span<CUevent> events) noexcept {
  for (CUevent& event : events) {
    if (event != nullptr) {
      (void)cuEventDestroy(event);
      event = nullptr;
    }
  }
}

}  // namespace

Submission::Submission(const Device& /*device*/) {
  // The device's primary context is current on this thread (device.h).
  // What is made is released in the reverse order if a later step throws.
  check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  try {
    check(cuStreamCreate(&image_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    std::array<CUevent, 1> origin{};
    make_events(origin, CU_EVENT_DEFAULT);
    origin_ = origin[0];
    // The starts and completions are timed (above, "Instants"); the
    // hand-offs order the streams only.
    make_events(starts_, CU_EVENT_DEFAULT);
    make_events(events_, CU_EVENT_DEFAULT);
    make_events(hand_offs_, CU_EVENT_DISABLE_TIMING);
    check(cuEventRecord(origin_, stream_), "cuEventRecord");
  } catch (...) {
    wait_idle();
    destroy_events(hand_offs_);
    destroy_events(events_);
    destroy_events(starts_);
    if (origin_ != nullptr) {
      (void)cuEventDestroy(origin_);
    }
    if (image_ != nullptr) {
      (void)cuStreamDestroy(image_);
    }
    (void)cuStreamDestroy(stream_);
    throw;
  }
  completed_.reserve(frames_in_flight);
}

Submission::~Submission() {
  wait_idle();
  destroy_events(hand_offs_);
  destroy_events(events_);
  destroy_events(starts_);
  (void)cuEventDestroy(origin_);
  (void)cuStreamDestroy(image_);
  (void)cuStreamDestroy(stream_);
}

frame::Seconds Submission::since_origin(CUevent event) const {
  float milliseconds = 0.0f;
  check(cuEventElapsedTime(&milliseconds, origin_, event),
        "cuEventElapsedTime");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): ms to s
  return frame::Seconds{static_cast<double>(milliseconds) / 1000.0};
}

void Submission::settle(std::uint32_t slot) {
  const std::optional<std::uint64_t> pending = frames_.pending(slot);
  if (!pending) {
    return;
  }
  check(cuEventSynchronize(events_.at(slot)),
        "frame " + std::to_string(*pending) + " on the GPU");
  // The start was recorded before the work, so it has completed too.
  completed_.push_back(Completed{*pending, since_origin(starts_.at(slot)),
                                 since_origin(events_.at(slot))});
  frames_.settled(slot);
}

FrameSlot Submission::begin(Parts parts) {
  const FrameSequence::Begun begun = frames_.begin(parts);
  if (begun.settle_first) {
    settle(begun.slot);
  }
  if (begun.after_image_half) {
    // A one-part frame after a two-part one: the light stream waits for
    // that frame's image half (frame_sequence.h).
    check(cuStreamWaitEvent(stream_, events_.at(*begun.after_image_half), 0),
          "cuStreamWaitEvent");
  }
  check(cuEventRecord(starts_.at(begun.slot), stream_), "cuEventRecord");
  return FrameSlot{stream_, begun.slot, begun.sequence, parts};
}

CUstream Submission::hand_off() {
  const std::uint32_t slot = frames_.hand_off();
  check(cuEventRecord(hand_offs_.at(slot), stream_), "cuEventRecord");
  check(cuStreamWaitEvent(image_, hand_offs_.at(slot), 0), "cuStreamWaitEvent");
  return image_;
}

void Submission::commit() {
  const FrameSequence::Committed committed = frames_.commit();
  const CUstream last = committed.parts == Parts::two ? image_ : stream_;
  check(cuEventRecord(events_.at(committed.slot), last), "cuEventRecord");
}

void Submission::wait_until_complete(std::uint64_t sequence) {
  frames_.check_committed(sequence);
  // Oldest first: frames complete in order (frame_sequence.h).
  while (const std::optional<std::uint32_t> slot =
             frames_.oldest_unsettled(sequence)) {
    settle(*slot);
  }
}

void Submission::finish() {
  if (frames_.next() > 0) {
    wait_until_complete(frames_.next() - 1);
  }
}

bool Submission::has_completed(std::uint64_t sequence) const noexcept {
  if (sequence >= frames_.next()) {
    return false;
  }
  if (frames_.is_settled(sequence)) {
    return true;
  }
  const auto slot = static_cast<std::uint32_t>(sequence % frames_in_flight);
  return frames_.pending(slot) == sequence &&
         cuEventQuery(events_.at(slot)) == CUDA_SUCCESS;
}

void Submission::take_completed(std::vector<Completed>& into) {
  into.insert(into.end(), completed_.begin(), completed_.end());
  completed_.clear();
}

void Submission::wait_idle() noexcept {
  if (image_ != nullptr) {
    (void)cuStreamSynchronize(image_);
  }
  if (stream_ != nullptr) {
    (void)cuStreamSynchronize(stream_);
  }
}

}  // namespace serenity::cuda
