#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/restir_pt/shift_replay.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's temporal reuse).
//
// Steps 5 to 7 of metal/passes/restir_pt/restir_pt.metal.h, four
// dispatches, each behind a barrier on the one before (GPU.8: a write and
// its reader):
//
//   restir_pt_temporal_shift   one thread a pixel: its temporal input's
//                              pixel found by reprojection and search; last
//                              frame's path there shifted here, and this
//                              pixel's path shifted there in last frame,
//                              where each reconnects at x2; the shifts that
//                              replay appended to a list.
//   restir_pt_replay_grid      one thread: the list's length as the next
//                              dispatch's grid (the spatial pass's kernel;
//                              the two passes share the list, its count and
//                              the grid, one after the other).
//   restir_pt_temporal_replay  one thread a replaying shift, an indirect
//                              dispatch whose grid stays on the device
//                              (GDSA.13).
//   restir_pt_temporal         one thread a pixel: the Talbot merge, into
//                              the reservoirs in place where a spatial reuse
//                              follows, else into the frame's final
//                              reservoirs and the light buffer.
//
// Reads: the scene at this frame's and last frame's transforms and glows,
// the structure (this frame's: a stated approximation, restir_pt.metal.h),
// the settings, the reuse constants, the path reservoirs, both frames'
// primary hits, the previous positions, last frame's final reservoirs and
// duplication map. Writes: the previous pixels, the shifted paths, the
// list, its count and grid, and the reservoirs (or the final reservoirs
// and the light buffer).
//
// Cost: two shifts a pixel whose surface was seen last frame, each read
// whole from the G-buffer, a reconnection's shadow ray or a replay; the
// merge's reads (restir_pt.metal.h counts them). Measured in docs/research/
// 2026-10-11-restir-pt-phase2.md.
class RestirPtTemporalPass {
 public:
  // Its pipelines specialized by the run's `settings`
  // (restir_pt/reuse_constants.h).
  RestirPtTemporalPass(const Library& library,
                       const contracts::PathReuseSettings& settings);

  // Records the pass. Throws MetalError if `resources` lacks any array,
  // structure or constant it reads or writes above.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  ShiftReplay reuse_;  // its four dispatches (restir_pt/shift_replay.h)
};

}  // namespace serenity::metal
