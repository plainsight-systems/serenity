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
// Steps 8 to 11 of metal/passes/restir_pt/restir_pt.metal.h, four
// dispatches, each behind a barrier on the one before (GPU.8: a write and
// its reader):
//
//   restir_pt_shift        one thread a pixel: each pixel's path shifted to
//                          its paired neighbors where it reconnects at x2;
//                          the pairs that replay appended to a list.
//   restir_pt_replay_grid  one thread: the list's length as the next
//                          dispatch's grid.
//   restir_pt_replay       one thread a replaying pair, an indirect dispatch
//                          whose grid stays on the device (GDSA.13).
//   restir_pt_spatial      one thread a pixel: the pairwise MIS merge, its
//                          estimate added to the light buffer.
//
// Reads: the scene, structure, settings, reuse textures, path reservoirs
// and primary hits. Writes: the shifted paths, the list, its count and
// grid, the light buffer and, with temporal reuse, the final reservoirs.
//
// Cost: three shifts a pixel, each its destination's 48-byte G-buffer
// record and a reconnection's shadow ray, more for the pairs that replay;
// the merge's reads (restir_pt.metal.h counts them). Measured in
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
  ShiftReplay reuse_;  // its four dispatches (restir_pt/shift_replay.h)
};

}  // namespace serenity::metal
