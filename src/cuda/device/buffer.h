#pragma once

#include <cuda.h>

#include <cstddef>

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

}  // namespace serenity::cuda
