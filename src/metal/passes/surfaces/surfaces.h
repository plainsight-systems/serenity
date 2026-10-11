#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's surfaces).
//
// ReSTIR DI's step 1 (metal/integrator/restir_di.metal.h): one thread a
// pixel (GDSA.8: work partitioned by output), the camera's path through a
// point in the pixel drawn anew each frame, as the path pass draws it
// (passes/path/path.h), to its first surface that aims at lights; written as
// the pixel's surface record and path state, and, in a graph with a
// temporal merge, its previous position (contracts/surface_record.h).
//
// Reads: the frame's constants and camera, the scene and this frame's
// structure; in a graph with a temporal merge, last frame's transforms
// (metal/scene/shape_transforms.h), to place the surface's point where its
// shape was. Writes: this frame's records, path states and previous
// positions (metal/frame/frame_images.h, ReuseImages), every pixel's,
// including those that reach no such surface (a record of kind none).
//
// Ordering: it writes this frame's half of the records' ping-pong, which
// the frame before this one's temporal merge read as last frame's. The
// renderer's queue barrier before the first pass of a frame that touches
// what frames share is before this one (metal/frame/renderer.h, GPU.8), so
// it never overwrites what the frame before still reads. The candidates'
// pass reads what it writes, after the renderer's dispatch barrier.
//
// Cost: per pixel, the camera segment's rays (one, and one more for each
// glass or mirror surface before the first rough one; more where a marble
// is in the way), its BSDF resolutions, and 128 bytes written, 140 with the
// previous position (contracts/surface_record.h). One dispatch.
//
// Optimization: the path's state is not kept in registers across a later
// pass's ray queries but written here and read by the shade
// (logical-overview.md, "Little is kept alive across a ray query"; GPU.3:
// what is cut is the state held across the queries, judged by the frame's
// time and the counters of docs/research/2026-10-10-path-kernel-counters.md,
// not by occupancy for its own sake).
class SurfacesPass {
public:
    explicit SurfacesPass(const Library& library);

    // Records the pass. Throws MetalError if `resources` has no scene, no
    // camera, no reuse constants, no records or no path states, or, when
    // `resources.previous_positions` is set, no last frame's transforms.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
