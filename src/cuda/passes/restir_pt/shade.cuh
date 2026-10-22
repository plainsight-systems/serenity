#pragma once

// Axis: Pass (ReSTIR PT's shade, step 12 of restir_pt.cuh, for one pixel).
//
// A traced pixel's light folded into the accumulated image, whose mean is
// the frame's radiance, as the path pass folds its own (passes/path/
// path.cu). With the checkerboard a pixel is folded in every other frame,
// so it starts over in either of a history's first two frames, the first
// that trace it (contract 20); its mean is of the frames that traced it.
//
// One function (F.1), called by the two kernels that shade: the shade
// pass's (restir_pt_shade/restir_pt_shade.cu), in a graph without the
// spatial pass, and the spatial merge's, which shades the pixel it has just
// merged, RTXDI's fused shading (restir_pt_spatial/
// restir_pt_spatial_kernels.cu; docs/research/2026-10-20-nvidia-hardware-
// design.md, section 8, change 5): the light then never makes a round trip
// through memory and no launch is spent on it (GDSA.6, GPU.6). The caller
// reports what it returns with count_non_finite() (film/accumulate.cuh).

#include "core/contracts/frame_constants.h"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/frame_launch.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"

namespace serenity {
namespace shaders {
namespace restir_pt {

// Folds `sample`, the light of `pixel` (inside the image), into the
// accumulated image; true if the sample is not finite, and so left out.
__device__ inline bool shade(const serenity::cuda::FrameLaunch& launch,
                             const Traced& traced, uint2 pixel, float3 sample) {
  // A pixel starts over in the image's first frame; with the checkerboard
  // in either of its first two, whichever traces it.
  const uint starting_frames = traced.checkerboard ? 2u : 1u;
  float4* const row = reinterpret_cast<float4*>(launch.accumulated.rows +
                                                std::uint64_t{pixel.y} *
                                                    launch.accumulated.pitch);
  row[pixel.x] = accumulate(row[pixel.x], sample,
                            launch.frame.accumulated_frames < starting_frames);
  return !finite(sample);
}

}  // namespace restir_pt
}  // namespace shaders
}  // namespace serenity
