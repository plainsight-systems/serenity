#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/restir_pt/shift_replay.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's paired spatial reuse).
//
// Steps 8 to 11 of metal/passes/restir_pt/restir_pt.metal.h, the
// sequence both reuses record (restir_pt/shift_replay.h):
//
//   restir_pt_pair           one thread a pixel: each pixel's partners
//                            paired, the pairs that only store stored, the
//                            pairs that reconnect at x2 and those that
//                            replay appended to a list each.
//   restir_pt_pair_grids     one thread: the lists' lengths as their grids.
//   restir_pt_spatial_shift  one thread a pair, over each list in turn: the
//                            pixel's path shifted to its partner.
//   restir_pt_spatial        one thread a pixel: the pairwise MIS merge, its
//                            estimate added to the pixel's light and shaded
//                            at once (step 12, restir_pt/shade.metal.h:
//                            RTXDI's fused shading; the schedule puts the
//                            shade straight after this pass, so the renderer
//                            records no shade pass beside it).
//
// Reads: the scene, structure, settings, reuse textures, path reservoirs
// and primary hits. Writes: the shifted paths, the lists, their counts
// and grids, the accumulated image, the count of samples not finite and,
// with temporal reuse, the final reservoirs.
//
// Cost: three shifts a pixel, each its destination's 48-byte G-buffer
// record and a reconnection's shadow ray, more for the pairs that replay,
// and a list's 4 bytes written and read; the merge's reads
// (restir_pt.metal.h counts them). Measured in
// docs/research/2026-10-11-restir-pt-phase2.md.
class RestirPtSpatialPass {
 public:
  // Its pipelines specialized by the run's `settings`
  // (restir_pt/reuse_constants.h).
  RestirPtSpatialPass(const Library& library,
                      const contracts::PathReuseSettings& settings);

  // Records the pass. Throws MetalError if `resources` lacks anything it
  // reads or writes above.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  ShiftReplay reuse_;  // its dispatches (restir_pt/shift_replay.h)
};

}  // namespace serenity::metal
