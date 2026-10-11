#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's spatial merge).
//
// ReSTIR DI's step 5 (metal/passes/restir_di/restir_di.metal.h, carried out by
// metal/reuse/spatial.metal.h): one thread a pixel; the reservoirs of k
// neighbors within the radius, similar in geometry and material, merged
// into the pixel's own by the rule of the graph's mode (core/passes/
// restir_di.h, bias_correction), its Z asked of each neighbor's surface in
// this frame.
//
// Reads: the scene, structure, camera, reuse constants and records, and
// the working reservoirs, at the pixel and its neighbors. Writes: this
// frame's final reservoirs, which the shade reads and the next frame's
// temporal merge reads as last frame's. Never in place: neighbors read the
// working array while it is written to the final one.
//
// Ordering: after the pass before it (the temporal merge's, or the
// candidates'), behind the renderer's dispatch barrier: every pixel's input
// is written before any neighbor reads it, which is why this is a pass of
// its own (logical-overview.md, principle 5). The final array it writes is
// this frame's half of the ping-pong; last frame's half, which the temporal
// merge read, it does not touch.
//
// Cost, per pixel at k neighbors: up to (1 + 2k) x 56 bytes read, 168 at
// RTXDI's k = 1, the neighbors' gathered within the radius (SIMD.5), and
// 24 written; at most 2k target evaluations, each with its light's 76
// bytes of logical loads; no shadow ray in basic mode, at most k in
// raytraced; 3k + 1 numbers (metal/reuse/spatial.metal.h counts them). Its
// time, against its count, is among the first measurements the
// implementation makes (GPU.10). One dispatch.
class RestirSpatialPass {
public:
    explicit RestirSpatialPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks any array or
    // structure it reads or writes above.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
