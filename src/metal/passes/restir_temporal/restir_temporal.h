#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's temporal merge).
//
// ReSTIR DI's step 4 (metal/passes/restir_di/restir_di.metal.h, carried out by
// metal/reuse/temporal.metal.h): one thread a pixel; last frame's final
// reservoir, where the pixel's surface was, merged into the candidates' by
// the rule of the graph's mode (core/contracts/reuse_settings.h,
// bias_correction),
// its normalization asked of last frame's surface and lights, and in
// raytraced mode of last frame's structure.
//
// Reads: this frame's scene, structure, camera, reuse constants, records,
// previous positions and the initial reservoirs (the working array);
// last frame's records and final reservoirs (metal/frame/frame_images.h,
// the other half of their ping-pongs), transforms and glows
// (metal/scene/shape_transforms.h, light_glows.h), and in raytraced mode
// its structure (metal/acceleration/scene_acceleration.h). Writes: one
// reservoir a pixel, in place in the working array when a spatial merge
// follows, which reads it at neighbors, else to this frame's final array
// (ReservoirOutput). In place is safe: each thread reads and writes its own
// pixel's alone. When the reuse constants say the frame starts its history
// over, each pixel's output is its initial reservoir, copied only where the
// output is not the input.
//
// Ordering: after the candidates' dispatch, behind the renderer's dispatch
// barrier. Last frame's arrays were written by the frame before, its final
// reservoirs last by its shade's write-back, which the renderer's queue
// barrier at the frame's first pass has waited for (metal/frame/
// renderer.h); this frame writes neither until the next frame.
//
// Cost, per pixel: 124 bytes read where the first pixel tried passes, up to
// 380 at a disocclusion, and 24 written (metal/reuse/temporal.metal.h
// counts them); at most two target evaluations (each with its light's 76
// bytes of logical loads, this frame's or last's); no shadow ray in basic
// mode, at most one in raytraced, against either frame's structure as the
// survivor's source says; two numbers, and the frame's two for the
// permutation. One dispatch.
class RestirTemporalPass {
public:
    RestirTemporalPass(const Library& library, ReservoirOutput output);

    // Records the pass. Throws MetalError if `resources` lacks any array or
    // structure it reads above, or the array `output` says it writes.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    ReservoirOutput output_;
};

}  // namespace serenity::metal
