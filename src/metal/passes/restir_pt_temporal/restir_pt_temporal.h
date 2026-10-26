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
// Steps 5 to 7 of metal/passes/restir_pt/restir_pt.metal.h, the
// sequence both reuses record (restir_pt/shift_replay.h):
//
//   restir_pt_temporal_pair   one thread a pixel: its temporal input's
//                             pixel found by reprojection and search; of
//                             its two shifts, last frame's path there to
//                             here and this pixel's path to there in last
//                             frame, each with no path stored as such, each
//                             that shifts appended to the reconnections' or
//                             the replays' list.
//   restir_pt_pair_grids      one thread: the lists' lengths as their grids
//                             (the spatial pass's kernel; the two passes
//                             share the lists, their counts and grids, one
//                             after the other).
//   restir_pt_temporal_shift  one thread a pair, over each list in turn.
//   restir_pt_temporal        one thread a pixel: the Talbot merge, into
//                             the reservoirs in place where a spatial reuse
//                             follows, else into the frame's final
//                             reservoirs and the light buffer.
//
// Reads: the scene at this frame's and last frame's transforms and glows,
// the structure (this frame's: a stated approximation, restir_pt.metal.h),
// the settings, the reuse constants, the path reservoirs, both frames'
// primary hits, the previous positions, last frame's final reservoirs and
// duplication map. Writes: the previous pixels, the shifted paths, the
// lists, their counts and grids, and the reservoirs (or the final reservoirs
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
  ShiftReplay reuse_;  // its dispatches (restir_pt/shift_replay.h)
};

}  // namespace serenity::metal
