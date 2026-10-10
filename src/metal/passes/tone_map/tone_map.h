#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/passes/tone_map.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (tone map).
//
// The frame's radiance as the look wants it shown: what is computed is the
// core's (core/passes/tone_map.h, steps 1 to 6), with its constants, which
// the shaders take from that header; this pass is how Metal computes it. The
// pass the window and movies end in (graphs/path.toml).
//
// How: four pipelines over the radiance image and the bloom pyramid
// (metal/frame/frame_images.h), in 12 dispatches:
//
//   - steps 1 and 2 for B_0: the first down pipeline, reading the radiance
//     image through the hardware's filter, exposing and clamping (step 1)
//     each of its 13 reads, writing B_0 as half floats;
//   - step 2 for B_1 .. B_5: the down pipeline, each reading the level
//     before;
//   - step 3 for B_4 .. B_0: the up pipeline, each reading the level below
//     and adding to its own, in place;
//   - steps 1 and 4 to 6: the finish pipeline at the frame's size, reading
//     the radiance image and B_0, writing the target.
//
// Bilinear reads through one sampler, clamp to edge, normalized coordinates
// at texel centers, as step 2 says; the radiance image's among them, which
// is RGBA32Float, so the pass needs a device that filters 32-bit floats
// (the M3 Max does), and refuses one that does not. Each dispatch reads
// what the one before wrote, so a barrier from dispatch to dispatch sits
// between each two: each a real hazard, the level just written and read
// next, at the dispatch stage alone (GPU.8). All twelve are recorded into
// the frame's one encoder (GPU.6). The radiance image was written by an earlier pass in the frame;
// the renderer records that barrier (metal/frame/renderer.h).
//
// The settings reach the shader in a 16-byte buffer of the pass's own
// (passes::ToneMap's shared layout), made at construction and never written
// again: the graph's, the same every frame.
//
// Cost, counted in passes over memory (GDSA.6), per frame, for P pixels: step
// 2 reads 13 texels per output pixel over P/4 + P/16 + ... (about P/3 pixels),
// 4.3 P reads; step 3, 9 per pixel over levels 0 to 4, about 3 P; step 4, 1
// full-size read and 9 of B_0, 10 P; 17 P reads in all, most of them of half
// floats at reduced size, and 12 dispatches with 11 barriers. What it costs
// a frame against the display pass, and the clamping of step 1 done on
// exact reads instead of filtered ones, which cost more and was not kept:
// docs/research/2026-10-09-pass-costs.md.
class ToneMapPass {
public:
    // Throws Error if `settings` are out of range (core/passes/tone_map.h),
    // the device cannot filter 32-bit floats, or it cannot make the
    // pipelines or the settings' buffer.
    ToneMapPass(const Device& device, const Library& library, Submission& submission,
                const passes::ToneMap& settings);

    // Records the pass. Throws Error if `resources` has no radiance image or
    // no bloom pyramid.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> down_first_;  // step 2, from the exposed radiance
    NS::SharedPtr<MTL::ComputePipelineState> down_;        // step 2, from a level
    NS::SharedPtr<MTL::ComputePipelineState> up_;          // step 3
    NS::SharedPtr<MTL::ComputePipelineState> finish_;      // steps 4 to 6
    NS::SharedPtr<MTL::Buffer> settings_;
    Resident resident_;
};

}  // namespace serenity::metal
