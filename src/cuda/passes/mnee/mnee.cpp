#include "cuda/passes/mnee/mnee.h"

#include <array>

#include "cuda/device/error.h"
#include "serenity/cuda_modules/mnee_kernels.h"
#include "serenity/optix_modules/mnee.h"

namespace serenity::cuda {

MneePass::MneePass(const OptixContext& context, TracedProgram::Records records)
    : solve_(context, optix_modules::mnee, "__raygen__mnee_solve", records),
      kernels_(cuda_modules::mnee_kernels),
      settle_(kernels_.kernel("mnee_settle")) {}

void MneePass::record(CUstream stream, CUdeviceptr launch,
                      const FrameLaunch& frame) const {
  if (frame.mnee_requests == 0 || frame.mnee_header == 0 ||
      frame.mnee_lost == 0 || frame.light == 0 || frame.non_finite == 0) {
    throw CudaError(
        "MneePass: the frame lacks its MNEE work list, its counter of lost "
        "requests or of samples not finite, or the light buffer it adds to");
  }
  // The list's capacity, one request a pixel (cuda/frame/mnee_list.h).
  solve_.launch(stream, launch, {frame.frame.width * frame.frame.height, 1});
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  cuda::launch(stream, settle_, LaunchShape{}, arguments);
}

}  // namespace serenity::cuda
