#include "cuda/device/buffer.h"

#include <utility>

#include "cuda/device/error.h"

namespace serenity::cuda {

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes == 0) {
    throw CudaError("DeviceBuffer: a buffer of no bytes");
  }
  check(cuMemAlloc(&address_, bytes), "cuMemAlloc");
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : address_(std::exchange(other.address_, 0)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    if (address_ != 0) {
      (void)cuMemFree(address_);
    }
    address_ = std::exchange(other.address_, 0);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() {
  if (address_ != 0) {
    (void)cuMemFree(address_);
  }
}

}  // namespace serenity::cuda
