#include "cuda/device/device.h"

#include <array>
#include <cstddef>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// The architecture the kernels are built for (cmake/CudaModules.cmake).
constexpr int built_major = 8;
constexpr int built_minor = 6;
// What cuDeviceGetName writes into: a name, its terminator, and room.
constexpr std::size_t name_capacity = 256;

}  // namespace

Device::Device() {
  check(cuInit(0), "cuInit");
  int count = 0;
  check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (count < 1) {
    throw CudaError("Device: the driver sees no CUDA device");
  }
  check(cuDeviceGet(&device_, 0), "cuDeviceGet");
  std::array<char, name_capacity> text{};
  check(cuDeviceGetName(text.data(), static_cast<int>(text.size()), device_),
        "cuDeviceGetName");
  int major = 0;
  int minor = 0;
  check(cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
        "cuDeviceGetAttribute");
  check(cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
        "cuDeviceGetAttribute");
  name_ = std::string(text.data()) + ", compute capability " +
          std::to_string(major) + "." + std::to_string(minor);
  if (major != built_major || minor != built_minor) {
    throw CudaError("Device: " + name_ +
                    "; the kernels are built for compute capability 8.6 "
                    "alone (cmake/CudaModules.cmake)");
  }
  check(cuDevicePrimaryCtxRetain(&context_, device_),
        "cuDevicePrimaryCtxRetain");
  if (const CUresult made = cuCtxSetCurrent(context_); made != CUDA_SUCCESS) {
    (void)cuDevicePrimaryCtxRelease(device_);
    check(made, "cuCtxSetCurrent");
  }
}

Device::~Device() { (void)cuDevicePrimaryCtxRelease(device_); }

}  // namespace serenity::cuda
