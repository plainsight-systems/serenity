#include "cuda/frame/mnee_list.h"

#include <array>
#include <cstdint>

#include "core/contracts/mnee_request.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

MneeList::MneeList(Submission& submission) : submission_(submission) {}

void MneeList::prepare(frame::Extent size) {
  if (size == size_) {
    return;
  }
  if (requests_.address() != 0) {
    submission_.finish();  // frames in flight may still read the old arrays
  }
  size_ = {};
  const std::size_t capacity =
      std::size_t{size.width} * size.height;  // one request a pixel
  requests_ = DeviceBuffer(capacity * sizeof(contracts::MneeRequest));
  header_ = DeviceBuffer(contracts::mnee_header_bytes);
  std::array<std::uint32_t, contracts::mnee_header_bytes / 4> words{};
  words[contracts::mnee_capacity_word] = static_cast<std::uint32_t>(capacity);
  upload_now(header_.address(), std::as_bytes(std::span(words)), "MneeList");
  // A walk a request at most, in each list.
  for (DeviceBuffer& list : walks_) {
    list = DeviceBuffer(capacity * sizeof(MneeWalk));
  }
  converged_ = DeviceBuffer(capacity * sizeof(MneeWalk));
  walk_header_ = DeviceBuffer(mnee_walk_header_words * sizeof(std::uint32_t));
  zero_now(walk_header_.address(), walk_header_.bytes(), "MneeList");
  size_ = size;
}

}  // namespace serenity::cuda
