#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's duplication map).
//
// Step 13 of metal/passes/restir_pt/restir_pt.metal.h: for each pixel, the
// share of the frame's final reservoirs in its 17 x 17 neighborhood that
// hold copies of its path (metal/reuse/duplication.metal.h), which the next
// frame's temporal reuse lowers its cap by. One dispatch of whole 24 x 24
// threadgroups (contracts::duplication_group), so every thread of a group
// loads its share of the group's keys (a group cut at the image's edge
// would leave some unloaded). With the checkerboard, over the packed grid
// of slots, its window 17 x 17 slots (restir_pt.metal.h, "Checkerboard
// rendering").
//
// Reads: the settings and the frame's final reservoirs. Writes: the
// duplication map.
//
// Cost: measured in docs/research/2026-10-11-restir-pt-phase2.md.
class RestirPtDuplicationPass {
public:
    // Throws MetalError if the device runs fewer than 24 x 24 threads in a
    // group of this kernel.
    explicit RestirPtDuplicationPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks anything it
    // reads or writes above.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
