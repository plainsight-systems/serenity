#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (depth of field: contract 21, core/contracts/depth_of_field.h).
//
// The scene's lens, applied to a denoised frame after the network
// (logical-overview.md, principle 12): the network's output, traced
// through a pinhole and sharp at every depth, blurred pixel by pixel by
// the circle of confusion the camera's thin lens gives the pixel's depth.
// Jimenez's depth of field ("Next Generation Post Processing in Call of
// Duty: Advanced Warfare", SIGGRAPH 2014), its structure kept and its
// steps each one dispatch, in this order, each after a barrier on the one
// before it (GPU.8: each reads what the one before wrote):
//
//   1. presort, at half the output's size each way (the contract's grid):
//      each half-resolution pixel the mean of its 2 x 2 block of the
//      network's radiance and the circle of confusion of the block's
//      nearest surface, in half-resolution pixels, signed (the contract);
//   2. tiles: each dof_tile x dof_tile tile's largest near-field radius,
//      Jimenez's tile max;
//   3. dilation: each tile the largest radius of the tiles within the
//      contract's `dilation` of it whose radius reaches it (a tile n tiles
//      away, n > 0, reaches with a radius of at least (n - 1) tiles),
//      Jimenez's 3 x 3 neighborhood grown to the largest radius the
//      output's size allows;
//   4. the gather, at half resolution: scatter as gather over Jimenez's
//      49-tap disc (the contract), its radius the larger of the pixel's own
//      and its tile's dilated near radius. Each tap is a pixel whose blur
//      may reach this one: its disc covers the center where its radius is
//      at least the tap's distance, by a pixel's soft edge
//      (saturate(r - d + 1/2), Jimenez's spread comparison), and it spreads
//      its light over that disc (a weight of 1 / max(r^2, 1/4), Jimenez's
//      sample alpha). The taps split in two fields:
//        - the far field, the center's own and every tap at or behind the
//          plane of focus: a tap behind the center (a larger signed
//          radius) reaches it with at most the center's own radius, so
//          the blurred background never spreads over a sharper surface in
//          front of it; normalized by its weights;
//        - the near field, every tap in front of the plane of focus, which
//          spreads over whatever is behind it, sharp or not: the bleed of
//          a defocused marble over the sharp table beside it. Its color
//          normalized by its weights, its alpha its coverage of the center,
//          the weights times the area a tap stands for, times Jimenez's
//          factor of 2, at most 1.
//      Jimenez splits the fields by depth against the tile's nearest; here
//      by the side of the plane of focus, the sign the circle of
//      confusion already carries, which needs no depth scale tuned to a
//      scene;
//   5. the median: a 3 x 3 median of both fields, channel by channel,
//      Jimenez's postfilter of the gather's undersampling;
//   6. the composite, at the output's size, in place: the network's own
//      pixel where its circle is under a pixel (sharp), the far field,
//      read bilinearly, where it is over two, blended between; the near
//      field over that by its alpha.
//
// Not taken from Jimenez: the 9-tap bilateral prefilter of the presort
// (the 2 x 2 mean in its place: the denoised input has no noise to
// prefilter), and his bokeh's shape (a disc here, as the thin lens's).
//
// What a frame gives it (FrameResources): the radiance image the denoise
// pass wrote at the output size, which it reads and writes; the full-size
// depth (contract 17), traced through the same pinhole; its constants
// (contract 21, filled from the camera's lens by core/passes/
// depth_of_field.h) and its images (metal/frame/depth_of_field_images.h).
// Throws MetalError, at record(), if any is missing. The renderer records
// nothing for a pinhole's frame (passes::blurs), whose image is the
// network's own.
//
// In place, the composite reads and writes each pixel of the radiance
// image by the same thread, after the presort, two barriers before it,
// has read the image whole: so no full-size image of its own (GPU.9: no
// 123 MB copy at 3456 x 2234), and the presenting pass and
// Renderer::read_accumulated() read the radiance image as before.
//
// Cost, at P output pixels: the presort reads 4 x 20 bytes a half pixel,
// 20 P bytes; the gather 49 taps of 8 bytes at P / 4 pixels, about 100 P
// bytes through the cache; the median 9 x 16 a half pixel; the composite
// 16 + 4 + 2 x 8 x 4 read and 16 written a pixel. Some 230 bytes a pixel,
// 1.8 GB at 3456 x 2234, most of it cached reads, no ray. Measured in
// docs/research/2026-10-18-production-sweep.md. The tiles and dilation are
// a thread a tile (GDSA.6: the passes counted; their work is the gather's
// 1 / 64th).
class DepthOfFieldPass {
public:
    explicit DepthOfFieldPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` lacks the radiance
    // image, the full-size depth, its constants or its images.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> presort_;
    NS::SharedPtr<MTL::ComputePipelineState> tiles_;
    NS::SharedPtr<MTL::ComputePipelineState> dilate_;
    NS::SharedPtr<MTL::ComputePipelineState> gather_;
    NS::SharedPtr<MTL::ComputePipelineState> median_;
    NS::SharedPtr<MTL::ComputePipelineState> composite_;
};

}  // namespace serenity::metal
