#pragma once

#include <cuda.h>

#include <cstddef>
#include <span>
#include <string_view>

#include "cuda/device/module.h"
#include "cuda/device/optix_context.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/traced_program.h"

namespace serenity::cuda {

// Axis: Pass (ReSTIR PT's reuse: the sequence both of its reuses record).
//
// ReSTIR PT's temporal and spatial reuse (restir_pt_temporal.h,
// restir_pt_spatial.h) are each one sequence, Metal's ShiftReplay
// (metal/passes/restir_pt/shift_replay.h), which the stream orders:
//
//   shift   a program, one thread a traced pixel (every pixel, or with the
//           checkerboard the frame's field, packed: FrameLaunch::traced):
//           the shifts that reconnect at x2 done, those that replay
//           appended to the replay list (restir_pt.cuh, replay_item());
//   replay  a program, one thread a place of the list, launched at its
//           capacity (contracts::max_path_neighbors a pixel, the core's
//           size, core/passes/restir_pt_arrays.h), a thread past the
//           list's count, which it reads where the shift appended, returning;
//   reset   the count set back to 0 on the stream, for the next list (the
//           spatial reuse's, after the temporal's, or the next frame's):
//           Metal's grid kernel's other half, with no grid to write;
//   merge   a kernel, one thread a traced pixel.
//
// Written once here for both (ES.3); each pass names its programs and
// kernel. Throws CudaError if OptiX refuses a program or the module cannot
// be loaded.
class ShiftReplay {
 public:
  // The names in a pass's modules (I.24: named, so no two pass in each
  // other's places).
  struct Names {
    std::string_view shift;   // a ray generation program
    std::string_view replay;  // a ray generation program
    std::string_view merge;   // a kernel
  };

  ShiftReplay(const OptixContext& context, std::span<const std::byte> programs,
              std::span<const std::byte> kernels, Names names,
              TracedProgram::Records records);

  // Records the sequence on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // the replay list or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  TracedProgram shift_;
  TracedProgram replay_;
  Module kernels_;
  CUfunction merge_ = nullptr;
};

}  // namespace serenity::cuda
