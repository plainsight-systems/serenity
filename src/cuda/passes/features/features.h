#pragma once

#include <cuda.h>

#include "cuda/device/optix_context.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/traced_program.h"

namespace serenity::cuda {

// Axis: Pass (the denoiser's features).
//
// Contract 17 (core/contracts/features.h), Metal's FeaturesPass
// (metal/passes/features/features.h, which says what it traces, writes and
// costs, and why a pass of its own): one program (features.cu), one thread
// a pixel, the camera's ray the light passes trace at the pixel and frame
// traced again, through glass and mirrors to the first surface that is
// neither; the four feature images (cuda/frame/feature_images.h) written.
//
// Throws CudaError if OptiX refuses the program.
class FeaturesPass {
 public:
  FeaturesPass(const OptixContext& context, TracedProgram::Records records);

  // Records the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // a feature image or the launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  TracedProgram program_;
};

}  // namespace serenity::cuda
