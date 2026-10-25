#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (the denoiser's features).
//
// Contract 17 (core/contracts/features.h), in two halves
// (docs/research/2026-10-23-apple-silicon-design.md, section 6, change 2):
//
//   rasterize()  a render pass, before the frame's compute encoder: the
//                scene's meshes drawn into the visibility buffer
//                (visibility.h; raster.metal), each pixel's nearest
//                triangle, its sample at the pixel's center plus the
//                frame's jitter where the frame takes it, by the pinhole
//                camera's projection; a draw a mesh, a still structure's
//                once, an instanced mesh's once a shape it places; depth
//                reversed, memoryless; nothing culled, as the tracer hits
//                a triangle from either side.
//   record()     one thread a pixel: the primary hit rebuilt from the
//                visibility buffer as trace() builds one (acceleration/
//                trace.metal.h, hit_of()); at it the view depth and the
//                motion since the frame before (last frame's camera and
//                transforms); through glass and mirrors, primary surface
//                replacement, traced as before, to the albedo and normal
//                of the first surface that is neither. Writes the four feature
//                images (feature_images.h), which only
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
// Cost, at P pixels: the meshes drawn once (some 1.7 million still
// triangles and each instanced mesh's, the marbles' fireflies most of
// them), 8 bytes a pixel written; then the hit rebuilt from one triangle,
// and at a delta primary surface the rays behind it, two branches of at most
// feature_delta_hops + 1 each (glass: in practice three, through a marble to
// the table, and one reflected); each surface's material resolved once (GPU.3:
// a few float3s live across the queries, nothing a path carries). Written, 28
// bytes a pixel (three RGBA16Float, one R32Float; GDSA.6): 14.5 MB at 960 x
// 540, 58 MB at 1920 x 1080. Read, the scene as the path pass reads it. One
// dispatch. Measured in docs/research/2026-10-14-denoiser-features.md.
class FeaturesPass {
 public:
  // Its pipelines, its depth test and its raster pass's argument table,
  // made on `device`.
  FeaturesPass(const Device& device, const Library& library);

  // Records the raster pass into `commands`, its own render encoder.
  // `after_readers`: the visibility buffer is shared by frames in flight,
  // so the pass first waits for the queue's earlier dispatches, which may
  // still read it (Metal 4 tracks no hazards; GPU.8); false where each
  // frame slot has its own. Throws MetalError if `resources` lacks the
  // camera, the frame's constants, the scene, the meshes or the raster
  // attachments.
  void rasterize(MTL4::CommandBuffer* commands, const FrameResources& resources,
                 bool after_readers) const;

  // Records the kernel, after the raster pass and a barrier from its
  // fragments (the renderer's). Throws MetalError if `resources` lacks the
  // scene, camera, transforms, glows, the feature constants, last frame's
  // transforms, any feature image, the visibility buffer or the meshes.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  // The raster pass's: a still structure's draw and an instanced mesh's,
  // the reversed depth test, and the argument table its draws bind.
  NS::SharedPtr<MTL::RenderPipelineState> still_;
  NS::SharedPtr<MTL::RenderPipelineState> instanced_;
  NS::SharedPtr<MTL::DepthStencilState> nearest_;
  NS::SharedPtr<MTL4::ArgumentTable> raster_arguments_;
};

}  // namespace serenity::metal
