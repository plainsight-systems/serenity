#pragma once

#include <cuda.h>

#include "cuda/device/module.h"
#include "cuda/device/optix_context.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/traced_program.h"

namespace serenity::cuda {

// Axis: Pass (manifold next event estimation's solve).
//
// The second half of MNEE (cuda/integrator/mnee.cuh), Metal's MneePass
// (metal/passes/mnee/mnee.h, which says why a list, what the solve costs and
// why its sums' order is the GPU's): the work list the initial pass's trees
// wrote (contract 16, core/contracts/mnee_request.h) solved, one thread a
// request, each request's light added to its pixel in the frame's light
// buffer. Recorded by ReSTIR PT's initial pass, right after its tree.
//
// Two launches, which the stream orders:
//
//   mnee_solve   the program (mnee.cu): a thread a place of the list, its
//                capacity's worth (one a pixel, cuda/frame/mnee_list.h), a
//                thread past the count returning: no launch takes its size
//                from the GPU (docs/research/2026-10-19-cuda-backend.md,
//                "Phase 4's design"), where Metal's grid kernel wrote it;
//   mnee_settle  one thread (mnee_kernels.cu): the requests that found the
//                list full added to the slot's count of lost requests, and
//                the count set back to 0 for the next frame.
//
// Throws CudaError if OptiX refuses the program or the module cannot be
// loaded.
class MneePass {
 public:
  MneePass(const OptixContext& context, TracedProgram::Records records);

  // Launches the solve and the settling on `stream`, the frame's launch
  // parameters at `launch`, `frame` their host copy. Throws CudaError if
  // the frame lacks MNEE's list or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  TracedProgram solve_;
  Module kernels_;
  CUfunction settle_ = nullptr;
};

}  // namespace serenity::cuda
