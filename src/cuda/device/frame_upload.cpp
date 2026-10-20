#include "cuda/device/frame_upload.h"

#include <algorithm>

namespace serenity::cuda {

namespace {

std::size_t stride_of(std::size_t size) {
  if (size == 0) {
    throw CudaError("FrameUpload: an array of no bytes");
  }
  return (size + FrameUpload::alignment - 1) / FrameUpload::alignment *
         FrameUpload::alignment;
}

}  // namespace

FrameUpload::FrameUpload(Submission& submission,
                         std::span<const std::byte> initial)
    : size_(initial.size()),
      stride_(stride_of(initial.size())),
      staging_(stride_ * frames_in_flight),
      device_(initial.size()) {
  for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    std::ranges::copy(initial, bytes(slot).begin());
  }
  // Before any frame: the device copy as given, whole before the
  // constructor returns.
  submission.finish();
  check(cuMemcpyHtoD(device_.address(), initial.data(), size_),
        "FrameUpload: cuMemcpyHtoD");
}

std::span<std::byte> FrameUpload::bytes(std::uint32_t slot) {
  if (slot >= frames_in_flight) {
    throw CudaError("FrameUpload: slot " + std::to_string(slot) +
                    " is not one of the " + std::to_string(frames_in_flight));
  }
  return staging_.bytes().subspan(std::size_t{slot} * stride_, size_);
}

void FrameUpload::upload(const FrameSlot& frame) {
  const std::span<const std::byte> copy = bytes(frame.slot);
  check(cuMemcpyHtoDAsync(device_.address(), copy.data(), size_, frame.stream),
        "FrameUpload: cuMemcpyHtoDAsync");
}

}  // namespace serenity::cuda
