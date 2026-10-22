#pragma once

#include <cuda.h>

#include "cuda/device/module.h"
#include "cuda/frame/frame_launch.h"

namespace serenity::cuda {

// Axis: Pass (ReSTIR PT's shade).
//
// Step 12 of cuda/passes/restir_pt/restir_pt.cuh, Metal's RestirPtShadePass
// (metal/passes/restir_pt_shade/restir_pt_shade.h): a kernel
// (restir_pt_shade.cu), one thread a traced pixel, the light buffer (and,
// with no reuse, the initial reservoir's own estimate) folded into the
// accumulated image, which is the frame's radiance, as the path pass's is.
// Recorded only in a graph without the spatial pass, whose merge shades
// (cuda/passes/restir_pt/shade.cuh).
//
// Throws CudaError if the module cannot be loaded.
class RestirPtShadePass {
 public:
  RestirPtShadePass();

  // Records the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // the light buffer, the accumulated image or its counter, or the launch
  // is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  Module module_;
  CUfunction kernel_ = nullptr;
};

}  // namespace serenity::cuda
