#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (the denoiser's features).
//
// Contract 17 (core/contracts/features.h), one thread a pixel: the camera's
// ray the light passes trace at the pixel and frame, traced again; at the
// primary surface its view depth and its motion since the frame before
// (last frame's camera and transforms); through glass and mirrors, primary
// surface replacement, to the albedo and normal of the first surface that
// is neither. Writes the four feature images (feature_images.h), which only
// the denoise pass reads, after the barrier the renderer records before it
// for the radiance it reads too: so the features pass needs no barrier of
// its own within the frame, and the renderer records none for it beyond the
// queue barrier before the first pass that writes what the frames in
// flight share (metal/frame/renderer.h).
//
// A pass of its own rather than a part of ReSTIR PT's initial pass
// (core/frame/schedule.h says why): a graph without it compiles, binds and
// dispatches what it did.
//
// Cost, at P pixels: one camera ray, and at a delta primary surface the
// rays behind it, two branches of at most feature_delta_hops + 1 each
// (glass: in practice three, through a marble to the table, and one
// reflected); each surface's material resolved once (GPU.3: a few float3s
// live across the queries, nothing a path carries). Written, 28 bytes a
// pixel (three RGBA16Float, one R32Float; GDSA.6): 14.5 MB at 960 x 540, 58
// MB at 1920 x 1080. Read, the scene as the path pass reads it. One
// dispatch. Measured in docs/research/2026-10-14-denoiser-features.md.
class FeaturesPass {
public:
    explicit FeaturesPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks the scene,
    // camera, transforms, glows, the feature constants, last frame's
    // transforms or any feature image.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
