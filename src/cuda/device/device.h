#pragma once

#include <cuda.h>

#include <string>

namespace serenity::cuda {

// Axis: GPU backend.
//
// The GPU, through CUDA's driver API (cuda.h), as Blender Cycles' CUDA
// device reaches it: device 0, its primary context retained and made
// current on the constructing thread, the one the backend runs on (CP.1),
// and released when this object goes (R.1). The machine it names, for every
// verification to say where it ran (docs/process/workflow.md).
//
// The kernels are built for one architecture, sm_86, the RTX 3090 Ti's
// (cmake/CudaModules.cmake), so a device of any other compute capability
// is refused here, by name, rather than by the first module that fails to
// load (I.5; P.7: catch run-time errors early).
//
// Throws CudaError if the driver cannot start, there is no device, or it is
// not compute capability 8.6.
class Device {
 public:
  Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  ~Device();

  CUdevice handle() const noexcept { return device_; }
  // The device's name and compute capability, e.g. "NVIDIA GeForce RTX 3090
  // Ti, compute capability 8.6".
  const std::string& name() const noexcept { return name_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  std::string name_;
};

}  // namespace serenity::cuda
