#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's initial pass).
//
// ReSTIR DI's steps 1 to 3 (metal/passes/restir_di/restir_di.metal.h, which
// its shader includes): one thread a pixel (GDSA.8: work partitioned by
// output). In order, in one kernel:
//
//   1. the camera's path through a point in the pixel drawn anew each
//      frame, as the path pass draws it (passes/path/path.h), to its first
//      surface that aims at lights; its record, path state and, in a graph
//      with a temporal merge, previous position written
//      (contracts/surface_record.h). The path state is written now, before
//      the candidates, so it is not alive across step 3's ray query.
//   2. the light candidates and the BSDF candidates at the record, weighed
//      by MIS over the two strategies, streamed into a reservoir, finished
//      as resampled importance sampling, W = w_sum / (M p̂), M both kinds;
//   3. the survivor's visibility traced, W set to 0 where it is hidden
//      (visibility reuse), and the reservoir written, its test this ray's,
//      here and now, its target the survivor's at the record.
//
// The record is packed as it is written (contracts/surface_record.h) and
// unpacked at once for steps 2 and 3, so the candidates are weighed at the
// surface every later pass reads back, not the full-precision one.
//
// One kernel, not two: steps 2 and 3 read nothing of another pixel's step
// 1, so splitting them would only write the record and read it back (32
// bytes a pixel) with a dispatch and a barrier between (GDSA.6, GPU.6).
// What is alive across visibility reuse's query is the streaming reservoir,
// the record and path state having been written before it, as it would be
// in a kernel of its own (restir_di.metal.h states it).
//
// Reads: the frame's constants and camera, the scene and this frame's
// structure, the reuse constants (M); in a graph with a temporal merge,
// last frame's transforms (metal/scene/shape_transforms.h), to place the
// surface's point where its shape was. Writes: this frame's records, path
// states and previous positions (metal/frame/frame_images.h, ReuseImages),
// every pixel's, including those that reach no such surface (a record
// whose bsdf is of kind none); and one reservoir a pixel, to the working
// array when a merge follows in the graph, else to this frame's final
// array (frame_resources.h, ReservoirOutput). A pixel with no surface, or
// in a scene with no light, writes an empty reservoir (no sample, W = 0,
// M = 0, target 0).
//
// Ordering: it writes this frame's half of the records' ping-pong, which
// the frame before this one's temporal merge read as last frame's. The
// renderer's queue barrier before the first pass of a frame that touches
// what frames share is before this one (metal/frame/renderer.h, GPU.8), so
// it never overwrites what the frame before still reads. The next ReSTIR
// pass reads what it writes, after the renderer's dispatch barrier.
//
// Cost, per pixel: the camera segment's rays (one, and one more for each
// glass or mirror surface before the first rough one) and its BSDF
// resolutions; for each light candidate a light chosen, a point drawn and
// a target evaluated (with its light's 76 bytes of logical loads), and with
// BSDF candidates a BSDF density; for each BSDF candidate a BSDF sample,
// one ray to the nearest surface, and where it strikes a light one target
// evaluation; one shadow ray; four numbers a candidate; 92 bytes written (32 record, 36 path state, 24 reservoir), 104
// with the previous position, nothing read back. One dispatch.
class RestirInitialPass {
public:
    RestirInitialPass(const Library& library, ReservoirOutput output);

    // Records the pass. Throws MetalError if `resources` has no scene, no
    // camera, no reuse constants, no records or path states, or no array
    // where `output` says it writes; or, when `resources.previous_positions`
    // is set, no last frame's transforms. The kernel writes the previous
    // positions exactly in a graph with a temporal merge, whose renderer
    // sets them.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    ReservoirOutput output_;
};

}  // namespace serenity::metal
