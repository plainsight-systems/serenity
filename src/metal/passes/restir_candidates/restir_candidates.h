#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's candidates).
//
// ReSTIR DI's steps 2 and 3 (metal/integrator/restir_di.metal.h): one
// thread a pixel; at the pixel's surface record, M candidates streamed into
// a reservoir, finished with Z = M, and the survivor's visibility traced,
// W set to 0 where it is hidden (visibility reuse).
//
// Reads: the scene and this frame's structure, the reuse constants (M), the
// records. Writes: one reservoir a pixel, to the working array when a merge
// follows in the graph, else to this frame's final array
// (frame_resources.h, ReservoirOutput; metal/frame/frame_images.h). A
// pixel with no surface (a record of kind none), or in a scene with no
// light, writes an empty reservoir (no sample, W = 0, M = 0).
//
// Ordering: after the surfaces pass's dispatch, behind the renderer's
// dispatch barrier; the next ReSTIR pass reads what it writes, behind
// another (GPU.8).
//
// Cost, per pixel: 64 bytes read, 16 written; M light choices, M points
// drawn and M target evaluations (target.metal.h); one shadow ray; 4M
// numbers. The ray is one query alive at a time, with the reservoir's 24
// bytes of registers across it, not the path's state (GPU.3). One
// dispatch.
class RestirCandidatesPass {
public:
    RestirCandidatesPass(const Library& library, ReservoirOutput output);

    // Records the pass. Throws MetalError if `resources` has no scene, no
    // reuse constants, no records, or no array where `output` says it
    // writes.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    ReservoirOutput output_;
};

}  // namespace serenity::metal
