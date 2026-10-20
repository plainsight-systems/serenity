#include "cuda/passes/checkerboard_fill/checkerboard_fill.h"

#include <array>

#include "core/contracts/checkerboard.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/checkerboard_fill.h"

namespace serenity::cuda {

CheckerboardFillPass::CheckerboardFillPass()
    : module_(cuda_modules::checkerboard_fill),
      kernel_(module_.kernel("checkerboard_fill")) {}

void CheckerboardFillPass::record(CUstream stream, CUdeviceptr launch,
                                  const FrameLaunch& frame) const {
  if (frame.accumulated.rows == 0 || frame.previous_light.rows == 0 ||
      frame.history_light.rows == 0 || frame.filled.rows == 0 ||
      frame.depth == 0 || frame.motion == 0) {
    throw CudaError(
        "CheckerboardFillPass: the frame lacks the accumulated image, the "
        "fill's history or image, or this frame's depth and motion");
  }
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  launch_per_pixel(
      stream, kernel_,
      {contracts::checkerboard_width(frame.frame.width), frame.frame.height},
      arguments);
}

}  // namespace serenity::cuda
