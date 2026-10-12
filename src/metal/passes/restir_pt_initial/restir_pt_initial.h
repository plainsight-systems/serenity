#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's initial sampling).
//
// Steps 1 to 4 of metal/passes/restir_pt/restir_pt.metal.h: the frame's
// light tiles, one thread an entry; a barrier (GPU.8: the tiles' write and
// the candidates that read them); then one thread a pixel, the path tree
// along the camera's ray resampled to one path.
//
// Reads: the scene, structure, camera, ReSTIR PT's settings and, with
// temporal reuse, last frame's transforms. Writes: the light tiles, and the
// path reservoirs, the primary hits (the G-buffer), the light buffer and,
// with temporal reuse, the previous positions, a pixel's each, which the
// passes after it read (renderer.h records the barrier).
//
// Cost: the tiles, 131072 lights placed and packed, 4 MB; the path tree,
// one path's rays (a ray and a shadow ray per vertex, as the path
// tracer's), and NEE's candidates, 32 at the primary hit and fewer after
// (metal/integrator/nee.metal.h), each a 32-byte tile entry read; 128
// bytes written a pixel, 140 with temporal reuse. Measured beside the path
// pass (docs/research/2026-10-11-restir-pt-phase2.md). Two dispatches.
class RestirPtInitialPass {
public:
    explicit RestirPtInitialPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks any array or
    // structure it reads or writes above.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> tiles_;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
