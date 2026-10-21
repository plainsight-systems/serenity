#pragma once

#include <cuda.h>

#include "cuda/device/optix_context.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/traced_program.h"
#include "cuda/passes/restir_pt/shift_replay.h"

namespace serenity::cuda {

// Axis: Pass (ReSTIR PT's paired spatial reuse).
//
// Steps 8 to 11 of cuda/passes/restir_pt/restir_pt.cuh, Metal's
// RestirPtSpatialPass (metal/passes/restir_pt_spatial/restir_pt_spatial.h,
// which gives what it reads, writes and costs): the shifts to the paired
// neighbors, the replays, the pairwise merge into the light buffer and, with
// temporal reuse, the final reservoirs. The sequence both reuses record
// (cuda/passes/restir_pt/shift_replay.h), with this pass's pairing and
// merge (restir_pt_spatial_kernels.cu) and its reconnections and replays
// (restir_pt_spatial.cu).
//
// Throws CudaError if OptiX refuses a program or the module cannot be
// loaded.
class RestirPtSpatialPass {
 public:
  RestirPtSpatialPass(const OptixContext& context,
                      TracedProgram::Records records);

  // Records the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // what the sequence reads or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const {
    sequence_.record(stream, launch, frame);
  }

 private:
  ShiftReplay sequence_;
};

}  // namespace serenity::cuda
