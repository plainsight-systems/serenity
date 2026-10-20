#pragma once

#include <cuda.h>

#include "core/frame/extent.h"
#include "cuda/device/optix_context.h"
#include "cuda/frame/traced_program.h"

namespace serenity::cuda {

// Axis: Pass (path tracing).
//
// The pass that runs the path tracer without reuse, Metal's PathPass
// (metal/passes/path/path.h, which says what it computes per pixel and what
// it costs): next event estimation weighed against BSDF sampling by MIS
// (cuda/integrator/path.cuh), once per pixel per frame, folded into the
// accumulated image (cuda/film/accumulate.cuh), so a still image converges
// toward the scene's full light transport.
//
// On CUDA it is one ray generation program (path.cu, cuda/frame/
// traced_program.h), one thread a pixel, reading the frame's launch
// parameters (cuda/frame/frame_launch.h): its constants, camera, scene,
// accumulated image and counter.
//
// Ordering: the pass reads the image the previous frame wrote; the one
// stream runs it after that frame's work, and every pass of a frame in
// order, so it needs no barrier, unlike Metal's encoder.
//
// Made once, outside the frame loop (MEM.9). Throws CudaError if OptiX
// refuses its program.
class PathPass {
 public:
  PathPass(const OptixContext& context, TracedProgram::Records records);

  // Launches the pass on `stream` over `size`, the frame's launch
  // parameters at `launch`. Throws CudaError if the launch is refused.
  void record(CUstream stream, CUdeviceptr launch, frame::Extent size) const;

 private:
  TracedProgram program_;
};

}  // namespace serenity::cuda
