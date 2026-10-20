#pragma once

#include <cuda.h>

#include "cuda/device/module.h"
#include "cuda/frame/frame_launch.h"

namespace serenity::cuda {

// Axis: Pass (ReSTIR PT's duplication map).
//
// Step 13 of cuda/passes/restir_pt/restir_pt.cuh, Metal's
// RestirPtDuplicationPass (metal/passes/restir_pt_duplication/
// restir_pt_duplication.h): a kernel (restir_pt_duplication.cu), whole
// blocks of 24 x 24 (contracts::duplication_group) over the arrays' grid
// (the image, or with the checkerboard the packed one), so every thread of
// a block loads its share of the block's keys into shared memory.
//
// Throws CudaError if the module cannot be loaded.
class RestirPtDuplicationPass {
 public:
  RestirPtDuplicationPass();

  // Records the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // the final reservoirs or the map, or the launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  Module module_;
  CUfunction kernel_ = nullptr;
};

}  // namespace serenity::cuda
