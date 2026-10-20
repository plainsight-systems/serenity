#pragma once

#include <cuda.h>

#include "cuda/device/module.h"
#include "cuda/device/optix_context.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/traced_program.h"
#include "cuda/passes/mnee/mnee.h"

namespace serenity::cuda {

// Axis: Pass (ReSTIR PT's initial sampling).
//
// Steps 1 to 4 of cuda/passes/restir_pt/restir_pt.cuh, Metal's
// RestirPtInitialPass (metal/passes/restir_pt_initial/restir_pt_initial.h,
// which gives what it reads and writes and what it costs), as four
// launches the stream orders:
//
//   restir_pt_light_power  a kernel, one block of light_power_threads
//                          (contracts/light_tile.h): the frame's power
//                          pyramid, its levels between __syncthreads;
//   restir_pt_light_tiles  a kernel, one thread an entry: the frame's light
//                          tiles, each drawn from the pyramid by power;
//   the tree               the program (restir_pt_initial.cu), one thread a
//                          traced pixel (every pixel, or with the
//                          checkerboard the frame's field, packed: contract
//                          20, FrameLaunch::traced): the path tree along the
//                          camera's ray resampled to one path;
//   MNEE                   the tree's requests solved, their light added to
//                          the light buffer (cuda/passes/mnee/mnee.h).
//
// Throws CudaError if OptiX refuses a program or a module cannot be loaded.
class RestirPtInitialPass {
 public:
  RestirPtInitialPass(const OptixContext& context,
                      TracedProgram::Records records);

  // Launches the pass on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // an array the pass reads or writes, or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  Module kernels_;
  CUfunction power_ = nullptr;
  CUfunction tiles_ = nullptr;
  TracedProgram tree_;
  MneePass mnee_;
};

}  // namespace serenity::cuda
