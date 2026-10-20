#include "cuda/device/optix_context.h"

#include <optix.h>
#include <optix_stubs.h>

#include <string>

#include "cuda/device/error.h"

// OptiX's function table, defined once in the program (optix_stubs.h's
// functions call through it); optixInit() fills it.
#include <optix_function_table_definition.h>

namespace serenity::cuda {

void check_optix(OptixResult result, std::string_view what,
                 std::string_view log) {
  if (result == OPTIX_SUCCESS) {
    return;
  }
  std::string message = std::string(what) + ": " + optixGetErrorName(result) +
                        " (" + optixGetErrorString(result) + ")";
  if (!log.empty()) {
    message += "\n" + std::string(log);
  }
  throw CudaError(message);
}

OptixContext::OptixContext(const Device& /*device*/) {
  check_optix(optixInit(), "optixInit");
  // The device's primary context is current on this thread (device.h); 0
  // names it.
  const OptixDeviceContextOptions options{};
  check_optix(optixDeviceContextCreate(nullptr, &options, &context_),
              "optixDeviceContextCreate");
}

OptixContext::~OptixContext() { (void)optixDeviceContextDestroy(context_); }

}  // namespace serenity::cuda
