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
// restir_pt_spatial.h) are each one sequence, which the stream orders. Its
// shifts run one thread a pixel and partner pair, compacted into lists
// before they run: ReSTIR PT Enhanced's pair lists (Lin et al. 2026, 6.2.2)
// on every pair that shifts, not only the replays
// (docs/research/2026-10-21-tracing-divergence.md, section 8). Metal's
// ShiftReplay (metal/passes/restir_pt/shift_replay.h) shifts from one
// thread a pixel; this is the 3090's.
//
//   pair       a kernel (it traces nothing), one thread a traced pixel
//              (every pixel, or with the checkerboard the frame's field,
//              packed: FrameLaunch::traced): each of its pairs that only
//              stores (not paired, or nothing held to shift) stored; each
//              that shifts appended to a list (restir_pt.cuh,
//              replay_item()), the pairs that reconnect at x2 to one and
//              those that replay to the other, so a warp of either list's
//              shifts does one kind of work (GDSA.13);
//   reconnect  a program, one thread a place of the reconnections' list;
//   replay     a program, one thread a place of the replays' list; each
//              launched at its list's capacity (contracts::max_path_neighbors
//              a pixel, the core's size, core/passes/restir_pt_arrays.h), a
//              thread past the list's count, which it reads where the
//              pairing appended, returning (no launch takes its size from
//              the GPU);
//   reset      both counts set back to 0 on the stream, for the next lists
//              (the spatial reuse's, after the temporal's, or the next
//              frame's);
//   merge      a kernel, one thread a traced pixel.
//
// A pair's shift is the same arithmetic on the same inputs whichever list
// holds it, written to its own slot: the shifted paths are the same to the
// bit as one thread a pixel's.
//
// Written once here for both (ES.3); each pass names its kernels and
// programs. Throws CudaError if OptiX refuses a program or the module
// cannot be loaded.
class ShiftReplay {
 public:
  // The names in a pass's modules (I.24: named, so no two pass in each
  // other's places).
  struct Names {
    std::string_view pair;       // a kernel
    std::string_view reconnect;  // a ray generation program
    std::string_view replay;     // a ray generation program
    std::string_view merge;      // a kernel
  };

  ShiftReplay(const OptixContext& context, std::span<const std::byte> programs,
              std::span<const std::byte> kernels, Names names,
              TracedProgram::Records records);

  // Records the sequence on `stream`, the frame's launch parameters at
  // `launch`, `frame` their host copy. Throws CudaError if the frame lacks
  // the shifted paths or either list, or a launch is refused.
  void record(CUstream stream, CUdeviceptr launch,
              const FrameLaunch& frame) const;

 private:
  TracedProgram reconnect_;
  TracedProgram replay_;
  Module kernels_;
  CUfunction pair_ = nullptr;
  CUfunction merge_ = nullptr;
};

}  // namespace serenity::cuda
