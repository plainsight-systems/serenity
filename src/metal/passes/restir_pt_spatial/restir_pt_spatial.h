#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's paired spatial reuse).
//
// Steps 5 to 8 of metal/passes/restir_pt/restir_pt.metal.h, four
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
// Reads: the scene, structure, camera, settings, reuse textures, path
// reservoirs and primary hits. Writes: the shifted paths, the list, its
// count and grid, and the light buffer.
//
// Cost: three shifts a pixel, each a primary ray traced again and a
// reconnection's shadow ray, more for the pairs that replay; the merge's
// reads (restir_pt.metal.h counts them). Measured in docs/research/
// 2026-10-11-restir-pt-phase1.md.
class RestirPtSpatialPass {
public:
    explicit RestirPtSpatialPass(const Library& library);

    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> shift_;
    NS::SharedPtr<MTL::ComputePipelineState> grid_;
    NS::SharedPtr<MTL::ComputePipelineState> replay_;
    NS::SharedPtr<MTL::ComputePipelineState> merge_;
};

}  // namespace serenity::metal
