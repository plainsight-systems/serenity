#pragma once

#include <optix_types.h>

#include <string_view>

#include "cuda/device/device.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// OptiX on the device: its function table, which optixInit() fills from the
// driver (OptiX's runtime is the driver's, pinned in cmake/toolchain.json),
// and its device context over the device's CUDA context, made once and
// released with this object (R.1), after every pipeline and structure made
// from it (their owners hold a reference and are released first, C.31).
// Every OptiX object of the backend is made through it.
//
// What OptiX reports at a failure is its error's name and, from the calls
// that compile or link, its log, which check_optix() puts in the thrown
// error. No callback is set: what a callback receives is printed while the
// run goes on (the guide, 4.1), and a failure here ends the run instead.
// Validation mode is off, OptiX's default (4.3: for debugging, since its checks
// reduce performance). The compile cache is OptiX's default, on (4.2;
// docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): it keeps
// OptiX's compile of the same input, so what runs is the same with it or
// without.
//
// Throws CudaError if OptiX cannot be loaded from the driver or the context
// cannot be made.
class OptixContext {
 public:
  explicit OptixContext(const Device& device);

  OptixContext(const OptixContext&) = delete;
  OptixContext& operator=(const OptixContext&) = delete;
  OptixContext(OptixContext&&) = delete;
  OptixContext& operator=(OptixContext&&) = delete;
  ~OptixContext();

  OptixDeviceContext handle() const noexcept { return context_; }

 private:
  OptixDeviceContext context_ = nullptr;
};

// Throws CudaError naming `what`, OptiX's name for `result` and `log`, the
// call's log where it has one, if `result` is not OPTIX_SUCCESS (I.10).
void check_optix(OptixResult result, std::string_view what,
                 std::string_view log = {});

}  // namespace serenity::cuda
