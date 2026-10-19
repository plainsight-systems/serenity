#include "metal/film/failure_count.h"

#include <span>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

FailureCount::FailureCount(const Device& device, Submission& submission,
                           std::string_view what)
    : submission_(submission), what_(what) {
  const auto pool = scoped_pool();
  counters_ = make_resident_buffer(
      device.handle(), submission, sizeof(std::uint32_t) * frames_in_flight,
      MTL::ResourceStorageModeShared, "the counts of " + what_, Zeroed::yes);
}

std::uint32_t* FailureCount::counter(std::uint32_t slot) noexcept {
  const std::span<std::uint32_t> counts(
      static_cast<std::uint32_t*>(counters_.buffer->contents()),
      frames_in_flight);
  return &counts[slot];
}

const std::uint32_t* FailureCount::counter(std::uint32_t slot) const noexcept {
  const std::span<const std::uint32_t> counts(
      static_cast<const std::uint32_t*>(counters_.buffer->contents()),
      frames_in_flight);
  return &counts[slot];
}

void FailureCount::begin_frame(std::uint32_t slot, std::uint64_t sequence) {
  if (slot >= frames_in_flight) {
    throw MetalError("FailureCount::begin_frame (" + what_ + "): no slot " +
                     std::to_string(slot));
  }
  // The slot's last frame has completed: begin() waited for it.
  if (counting_.at(slot) != 0) {
    total_ += *counter(slot);
  }
  *counter(slot) = 0;
  counting_.at(slot) = sequence + 1;
}

MTL::GPUAddress FailureCount::address(std::uint32_t slot) const {
  if (slot >= frames_in_flight) {
    throw MetalError("FailureCount::address (" + what_ + "): no slot " +
                     std::to_string(slot));
  }
  return counters_.address() + sizeof(std::uint32_t) * slot;
}

std::uint64_t FailureCount::count() const noexcept {
  std::uint64_t total = total_;
  std::uint32_t slot = 0;
  for (const std::uint64_t counted : counting_) {
    if (counted != 0 && submission_.has_completed(counted - 1)) {
      total += *counter(slot);
    }
    ++slot;
  }
  return total;
}

}  // namespace serenity::metal
