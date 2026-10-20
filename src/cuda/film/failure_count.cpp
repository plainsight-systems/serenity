#include "cuda/film/failure_count.h"

#include <cstring>

#include "cuda/device/error.h"

namespace serenity::cuda {

FailureCount::FailureCount(Submission& submission, std::string_view what)
    : submission_(submission),
      what_(what),
      counters_(sizeof(std::uint32_t) * frames_in_flight),
      copies_(sizeof(std::uint32_t) * frames_in_flight) {}

std::uint32_t FailureCount::host_count(std::uint32_t slot) const noexcept {
  std::uint32_t count = 0;
  std::memcpy(&count,
              copies_.bytes().subspan(sizeof(std::uint32_t) * slot).data(),
              sizeof(count));
  return count;
}

CUdeviceptr FailureCount::address(std::uint32_t slot) const {
  if (slot >= frames_in_flight) {
    throw CudaError("FailureCount (" + what_ + "): no slot " +
                    std::to_string(slot));
  }
  return counters_.address() + sizeof(std::uint32_t) * slot;
}

void FailureCount::begin_frame(const FrameSlot& frame) {
  const CUdeviceptr counter = address(frame.slot);
  // The slot's last frame has completed, and its copy back with it.
  if (counting_.at(frame.slot) != 0) {
    total_ += host_count(frame.slot);
  }
  counting_.at(frame.slot) = frame.sequence + 1;
  check(cuMemsetD32Async(counter, 0, 1, frame.stream),
        "FailureCount: cuMemsetD32Async");
}

void FailureCount::end_frame(const FrameSlot& frame) {
  const CUdeviceptr counter = address(frame.slot);
  check(cuMemcpyDtoHAsync(
            copies_.bytes().subspan(sizeof(std::uint32_t) * frame.slot).data(),
            counter, sizeof(std::uint32_t), frame.stream),
        "FailureCount: cuMemcpyDtoHAsync");
}

std::uint64_t FailureCount::count() const noexcept {
  std::uint64_t total = total_;
  std::uint32_t slot = 0;
  for (const std::uint64_t counted : counting_) {
    if (counted != 0 && submission_.has_completed(counted - 1)) {
      total += host_count(slot);
    }
    ++slot;
  }
  return total;
}

}  // namespace serenity::cuda
