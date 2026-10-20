#include "cuda/passes/restir_pt_shade/restir_pt_shade.h"

#include <array>

#include "cuda/device/error.h"
#include "serenity/cuda_modules/restir_pt_shade.h"

namespace serenity::cuda {

RestirPtShadePass::RestirPtShadePass()
    : module_(cuda_modules::restir_pt_shade),
      kernel_(module_.kernel("restir_pt_shade")) {}

void RestirPtShadePass::record(CUstream stream, CUdeviceptr launch,
                               const FrameLaunch& frame) const {
  if (frame.light == 0 || frame.path_reservoirs == 0 ||
      frame.accumulated.rows == 0 || frame.non_finite == 0) {
    throw CudaError(
        "RestirPtShadePass: the frame lacks the light buffer, the path "
        "reservoirs, the accumulated image or its counter");
  }
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  launch_per_pixel(stream, kernel_, {frame.traced.x, frame.traced.y},
                   arguments);
}

}  // namespace serenity::cuda
