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
// wrote (contract 16, core/contracts/mnee_request.h) solved, each request's
// light added to its pixel in the frame's light buffer. Recorded by ReSTIR
// PT's initial pass, right after its tree.
//
// The Newton walks in rounds (docs/research/2026-10-21-tracing-
// divergence.md, section 4): measured on the RTX 3090 Ti, a fifth of the
// walks fail and spend 57% of the iterations, and a warp of 32 waits on
// its longest, its lanes busy 18% of the solve's instructions. Taken in
// rounds of mnee_round_iterations (contract 16, core/contracts/mnee_request.h),
// the walks still going gathered into full warps between rounds, each walk
// takes the same iterations as one taken whole. The launches, which the stream
// orders, each at its list's capacity (one a pixel, cuda/frame/
// mnee_list.h), a thread past the count returning:
//
//   mnee_walk_start   a kernel (mnee_kernels.cu): each request's walk
//                     begun and taken for a round, into round list 0;
//   mnee_walk_round   a kernel, mnee_rounds - 1 times: a round list's walks
//                     taken for another round, into the other, whose count
//                     is set to 0 on the stream first;
//   mnee_finish       the program (mnee.cu): each converged walk's light,
//                     its chain's visibility traced;
//   mnee_settle       one thread: the requests that found the list full
//                     added to the slot's count of lost requests, and every
//                     count set back to 0 for the next frame.
//
// Throws CudaError if OptiX refuses the program or the module cannot be
// loaded.
class MneePass {
 public:
  MneePass(const OptixContext& context, TracedProgram::Records records);

  // Launches the walks, the finish and the settling on `stream`, the
  // frame's launch parameters at `launch`, `frame` their host copy. Throws
  // CudaError if the frame lacks MNEE's lists or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  TracedProgram finish_;
  Module kernels_;
  CUfunction start_ = nullptr;
  CUfunction round_ = nullptr;
  CUfunction settle_ = nullptr;
};

}  // namespace serenity::cuda
