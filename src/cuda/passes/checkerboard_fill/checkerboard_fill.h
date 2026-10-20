#pragma once

#include <cuda.h>

#include "cuda/device/module.h"
#include "cuda/frame/frame_launch.h"

namespace serenity::cuda {

// Axis: Pass (the checkerboard's fill).
//
// With checkerboard rendering (contract 20), Metal's CheckerboardFillPass
// (metal/passes/checkerboard_fill/checkerboard_fill.h, which states the
// fill and the alternatives measured): a kernel (checkerboard_fill.cu), one
// thread a slot of the packed grid, the light handed on written whole, a
// traced pixel's from the accumulated image and an untraced one's
// reprojected from the frame before's history or from its traced
// neighbors; and this frame's history, its light and depth, for the next
// frame's fill. In the data set's input graph, the one with no network,
// the image handed on is the one the presenting pass shows and the
// headless renderer writes (cuda/frame/checkerboard_images.h).
//
// Throws CudaError if the module cannot be loaded.
class CheckerboardFillPass {
 public:
  CheckerboardFillPass();

  // Records the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // an image the fill reads or writes, or the launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  Module module_;
  CUfunction kernel_ = nullptr;
};

}  // namespace serenity::cuda
