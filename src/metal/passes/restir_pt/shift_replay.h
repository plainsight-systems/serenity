#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <string_view>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's reuse: the sequence both of its reuses record).
//
// ReSTIR PT's temporal and spatial reuse (restir_pt_temporal.h,
// restir_pt_spatial.h) each pair their pixels, then shift one thread a pair
// of a list: ReSTIR PT Enhanced's pair lists (Lin et al. 2026, 6.2.2) on
// every pair that shifts, as on the RTX 3090 Ti (cuda/passes/restir_pt/
// shift_replay.h; docs/research/2026-10-21-tracing-divergence.md, section
// 8), so a SIMD group's shifts do one kind of work (GPU.4, GDSA.13). Each
// dispatch behind a barrier on the one before (GPU.8: a write and its
// reader), but the two shifts:
//
//   pair     one thread a traced pixel (every pixel, or with the
//            checkerboard the frame's field: frame_resources.h, traced;
//            contract 20): each of its pairs that only stores (not paired,
//            or nothing held to shift) stored; each that shifts appended to
//            a list (restir_pt.metal.h, replay_item()), the pairs that
//            reconnect at x2 to one and those that replay to the other
//            (contracts::pair_lists), a SIMD group's at once (GDSA.10);
//   grid     one thread: both lists' counts as their indirect grids
//            (restir_pt_pair_grids, metal/device/indirect_grid.h), their
//            threadgroups contract 15's replay_group, and both counts set
//            back to 0 for the next lists (the spatial reuse's, after the
//            temporal's, or the next frame's);
//   shift    one pipeline, one thread a pair, dispatched by each list's
//            grid in turn, the list and its grid rebound between (Metal 4
//            takes the argument table's bindings at each dispatch): the
//            reconnections, then the replays, with no barrier between, for
//            they write different slots and read the same inputs, so the GPU
//            may run them together;
//   merge    one thread a traced pixel.
//
// A pair's shift is the same arithmetic on the same inputs whichever list
// holds it, written to its own slot.
//
// Written once here for both (ES.3); each pass names its kernels and binds
// its own resources before recording it. Throws MetalError, naming the
// pass, if the shift's pipeline takes fewer threads a group than the
// grids give it.
class ShiftReplay {
 public:
  // The names in the library of a pass's kernels (I.24: named, so no two
  // pass in each other's places).
  struct Names {
    const char* pair;
    const char* shift;
    const char* merge;
  };

  // The pairing, shift and merge specialized by the run's `settings`
  // (reuse_constants.h).
  ShiftReplay(const Library& library,
              const contracts::PathReuseSettings& settings, Names names,
              std::string_view pass);

  // The dispatches, the pairing and the merge over `resources.traced`, the
  // shifts over the lists `resources` holds. Throws MetalError, naming the
  // pass, if it lacks the lists, their counts or their grids.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> pair_;
  NS::SharedPtr<MTL::ComputePipelineState> grid_;
  NS::SharedPtr<MTL::ComputePipelineState> shift_;
  NS::SharedPtr<MTL::ComputePipelineState> merge_;
  std::string_view pass_;
};

}  // namespace serenity::metal
