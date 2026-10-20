#include "cuda/device/buffer.h"

#include <cstring>
#include <string>
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

PinnedBuffer::PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes == 0) {
    throw CudaError("PinnedBuffer: a buffer of no bytes");
  }
  void* memory = nullptr;
  check(cuMemHostAlloc(&memory, bytes, 0), "cuMemHostAlloc");
  data_ = static_cast<std::byte*>(memory);
  std::memset(data_, 0, bytes_);
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      (void)cuMemFreeHost(data_);
    }
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

PinnedBuffer::~PinnedBuffer() {
  if (data_ != nullptr) {
    (void)cuMemFreeHost(data_);
  }
}

void upload_now(CUdeviceptr target, std::span<const std::byte> bytes,
                std::string_view what) {
  check(cuMemcpyHtoD(target, bytes.data(), bytes.size()),
        std::string(what) + ": cuMemcpyHtoD");
  check(cuCtxSynchronize(), std::string(what) + ": cuCtxSynchronize");
}

void zero_now(CUdeviceptr target, std::size_t bytes, std::string_view what) {
  check(cuMemsetD8(target, 0, bytes), std::string(what) + ": cuMemsetD8");
  check(cuCtxSynchronize(), std::string(what) + ": cuCtxSynchronize");
}

}  // namespace serenity::cuda
