#include "cuda/passes/restir_pt_duplication/restir_pt_duplication.h"

#include <array>

#include "core/contracts/path_reuse_settings.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/restir_pt_duplication.h"

namespace serenity::cuda {

RestirPtDuplicationPass::RestirPtDuplicationPass()
    : module_(cuda_modules::restir_pt_duplication),
      kernel_(module_.kernel("restir_pt_duplication")) {}

void RestirPtDuplicationPass::record(CUstream stream, CUdeviceptr launch,
                                     const FrameLaunch& frame) const {
  if (frame.path_finals == 0 || frame.duplication == 0) {
    throw CudaError(
        "RestirPtDuplicationPass: the frame lacks the final reservoirs or the "
        "duplication map");
  }
  constexpr unsigned group = contracts::duplication_group;
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  cuda::launch(stream, kernel_,
               LaunchShape{
                   .blocks_x = (frame.traced.x + group - 1) / group,
                   .blocks_y = (frame.traced.y + group - 1) / group,
                   .threads_x = group,
                   .threads_y = group,
               },
               arguments);
}

}  // namespace serenity::cuda
