#pragma once

// Axis: Pass (ReSTIR PT's shade, step 12 of restir_pt.metal.h, for one
// pixel).
//
// A traced pixel's light folded into the accumulated image, whose mean is
// the frame's radiance, as the path pass folds its own (passes/path/
// path.metal). With the checkerboard a pixel is folded in every other
// frame, so it starts over in either of a history's first two frames, the
// first that trace it (contract 20); its mean is of the frames that traced
// it.
//
// One function (F.1), called by the two kernels that shade: the shade
// pass's (restir_pt_shade/restir_pt_shade.metal), in a graph without the
// spatial pass, and the spatial merge's, which shades the pixel it has just
// merged, RTXDI's fused shading (restir_pt_spatial/restir_pt_spatial.metal;
// docs/research/2026-10-23-apple-silicon-design.md, section 6, change 5;
// CUDA's passes/restir_pt/shade.cuh): the light then makes no round trip
// through memory and no dispatch is spent on it (GDSA.6, GPU.6). The caller
// reports what it returns with count_non_finite() (film/accumulate.metal.h).

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "metal/film/accumulate.metal.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"

namespace serenity {
namespace shaders {
namespace restir_pt {

// Folds `sample`, the light of `pixel` (inside the image), into
// `accumulated`; true if the sample is not finite, and so left out.
inline bool shade(
    metal::texture2d<float, metal::access::read_write> accumulated,
    constant serenity::contracts::FrameConstants& frame, Traced traced,
    uint2 pixel, float3 sample) {
  // A pixel starts over in the image's first frame; with the checkerboard
  // in either of its first two, whichever traces it.
  const uint starting_frames = traced.checkerboard ? 2u : 1u;
  accumulated.write(accumulate(accumulated.read(pixel), sample,
                               frame.accumulated_frames < starting_frames),
                    pixel);
  return !finite(sample);
}

}  // namespace restir_pt
}  // namespace shaders
}  // namespace serenity
