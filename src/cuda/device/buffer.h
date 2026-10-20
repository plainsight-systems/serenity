#pragma once

#include <cuda.h>

#include <cstddef>
#include <span>

namespace serenity::cuda {

// Axis: GPU backend.
//
// Device memory, owned (R.1, E.6): `bytes` of it allocated at construction
// and freed with this object, which must not outlive the device's context
// nor be freed while the GPU still uses it: its owners are released after
// the submission has waited for every frame in flight (submission.h,
// Lifetime). Made outside the frame loop and kept (MEM.9). Move-only, the
// moved-from buffer empty (address 0, 0 bytes).
//
// Throws CudaError if the device cannot allocate it, or for 0 bytes, which
// the driver refuses.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  CUdeviceptr address() const noexcept { return address_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  CUdeviceptr address_ = 0;
  std::size_t bytes_ = 0;
};

// Axis: GPU backend.
//
// Pinned host memory, owned (R.1): `bytes` of page-locked memory, which the
// GPU's copy engines read and write directly, so a copy from it on a stream
// runs when the stream reaches it, not when it is issued (the CUDA Best
// Practices Guide, "Pinned memory"; GPU.7: overlap needs it). Where a
// frame's data is staged for its upload (frame_upload.h) and where a
// frame's counts land on their way back (cuda/film/failure_count.h). Freed
// with this object, under the same lifetime rule as DeviceBuffer's. Its
// bytes are zeroed at construction. Move-only, the moved-from buffer empty.
//
// Throws CudaError if the driver cannot allocate it, or for 0 bytes.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t bytes);

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
  ~PinnedBuffer();

  std::span<std::byte> bytes() noexcept { return {data_, bytes_}; }
  std::span<const std::byte> bytes() const noexcept { return {data_, bytes_}; }

 private:
  std::byte* data_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace serenity::cuda
