#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

#include "core/frame/extent.h"
#include "cuda/acceleration/structure_block.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/pipeline.h"

namespace serenity::cuda {

// Axis: Frame graph (a pass's program that traces).
//
// One ray generation program of a pass's OptiX module, as every tracing
// pass of the backend makes it (ES.3: written once): its pipeline
// (cuda/device/pipeline.h) with the backend's tracing contract
// (cuda/acceleration/structure_block.h): the triangles' hit group, OptiX's
// own, and MNEE's seed's intersection program, which every tracing module
// defines (traced_module.cuh), a hit record each, the payload
// words a trace carries, and the frame's launch parameters in `launch`
// (frame_launch.h). Launched over a grid, a thread a pixel or a list's item,
// with the frame's launch parameters at the address the renderer uploaded
// them to.
//
// Made once, outside the frame loop (MEM.9). Throws CudaError if OptiX
// refuses the program.
class TracedProgram {
 public:
  using Records = std::span<const Pipeline::HitRecord, structure_records>;

  TracedProgram(const OptixContext& context, std::span<const std::byte> module,
                std::string_view ray_generation, Records records);

  // Launches the program over `grid` on `stream`, the frame's launch
  // parameters at `launch`. Throws CudaError if the launch is refused.
  void launch(CUstream stream, CUdeviceptr launch, frame::Extent grid) const;

 private:
  Pipeline pipeline_;
};

}  // namespace serenity::cuda
