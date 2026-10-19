#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <string_view>

#include "core/frame/extent.h"
#include "metal/device/library.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's reuse: the sequence both of its reuses record).
//
// ReSTIR PT's temporal and spatial reuse (restir_pt_temporal.h,
// restir_pt_spatial.h) are each four dispatches in one order, each behind
// a barrier on the one before (GPU.8: a write and its reader):
//
//   shift   one thread a traced pixel (every pixel, or with the
//           checkerboard the frame's field: frame_resources.h, traced;
//           contract 20): the shifts that reconnect at x2 done, those
//           that replay appended to the replay list (restir_pt.metal.h,
//           replay_item());
//   grid    one thread: the list's length as the replay's indirect grid
//           (restir_pt_replay_grid, metal/device/indirect_grid.h), its
//           threadgroups contract 15's replay_group;
//   replay  one thread a replaying shift, dispatched by that grid;
//   merge   one thread a traced pixel.
//
// Written once here for both (ES.3); each pass names its kernels and binds
// its own resources before recording it. Throws MetalError, naming the
// pass, if the replay's pipeline takes fewer threads a group than the
// grid gives it.
class ShiftReplay {
 public:
  ShiftReplay(const Library& library, const char* shift, const char* replay,
              const char* merge, std::string_view pass);

  // The four dispatches, the shift and the merge over the grid `traced`
  // (frame_resources.h), the replay by the grid at `replay_grid`.
  void record(MTL4::ComputeCommandEncoder* encoder, frame::Extent traced,
              MTL::GPUAddress replay_grid) const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> shift_;
  NS::SharedPtr<MTL::ComputePipelineState> grid_;
  NS::SharedPtr<MTL::ComputePipelineState> replay_;
  NS::SharedPtr<MTL::ComputePipelineState> merge_;
};

}  // namespace serenity::metal
