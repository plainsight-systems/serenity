#include "cuda/device/offscreen.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

std::size_t rgba_bytes(frame::Extent size) {
  if (size.width == 0 || size.height == 0) {
    throw CudaError("Offscreen: an image of no pixels");
  }
  return std::size_t{size.width} * size.height * 4;
}

}  // namespace

Offscreen::Offscreen(Submission& submission, frame::Extent size)
    : submission_(submission), size_(size), image_(rgba_bytes(size)) {}

std::size_t Offscreen::rgba_size() const noexcept {
  return std::size_t{size_.width} * size_.height * 4;
}

void Offscreen::read_rgba(std::span<std::uint8_t> out) const {
  if (out.size() != rgba_size()) {
    throw CudaError("Offscreen::read_rgba: " + std::to_string(out.size()) +
                    " bytes for an image of " + std::to_string(rgba_size()));
  }
  submission_.finish();
  check(cuMemcpyDtoH(out.data(), image_.address(), out.size()), "cuMemcpyDtoH");
}

}  // namespace serenity::cuda
