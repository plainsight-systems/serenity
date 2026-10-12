#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's initial sampling).
//
// Steps 1 to 4 of metal/passes/restir_pt/restir_pt.metal.h: one thread a
// pixel, the path tree along the camera's ray resampled to one path.
//
// Reads: the scene, structure, camera and ReSTIR PT's settings. Writes: the
// path reservoirs, the primary hits and the light buffer, a pixel's each,
// which the passes after it read (renderer.h records the barrier).
//
// Cost: the path tree, one path's rays (a ray and a shadow ray per vertex,
// as the path tracer's), and NEE's candidates, 32 at the primary hit and
// fewer after (metal/integrator/nee.metal.h); 88 bytes written a pixel.
// Measured beside the path pass (docs/research/2026-10-11-restir-pt-
// phase1.md). One dispatch.
class RestirPtInitialPass {
public:
    explicit RestirPtInitialPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks any array or
    // structure it reads or writes above.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
