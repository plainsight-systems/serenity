#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/tone_map.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (tone map).
//
// The frame's radiance as the look wants it shown: what is computed is the
// core's (core/frame/tone_map.h, steps 1 to 6), with its constants, which
// the shaders take from that header; this pass is how Metal computes it. The
// pass the window and movies end in (graphs/path.toml).
//
// How: four pipelines over the radiance image and the bloom pyramid
// (metal/frame/frame_images.h), in 12 dispatches:
//
//   - steps 1 and 2 for B_0: the first down pipeline, reading the radiance
//     image, exposing and clamping it (step 1) at each of its 13 reads,
//     writing B_0 as half floats;
//   - step 2 for B_1 .. B_5: the down pipeline, each reading the level
//     before;
//   - step 3 for B_4 .. B_0: the up pipeline, each reading the level below
//     and adding to its own, in place;
//   - steps 1 and 4 to 6: the finish pipeline at the frame's size, reading
//     the radiance image and B_0, writing the target.
//
// Bilinear reads through one sampler, clamp to edge, normalized coordinates
// at texel centers, as step 2 says. Each dispatch reads what the one before
// wrote, so a barrier from dispatch to dispatch sits between each two
// (GPU.7). The radiance image was written by an earlier pass in the frame;
// the renderer records that barrier (metal/frame/renderer.h).
//
// The settings reach the shader in a 16-byte buffer of the pass's own
// (frame::ToneMap's shared layout), made at construction and never written
// again: the graph's, the same every frame.
//
// Cost, per frame, for P pixels: step 2 reads 13 texels per output pixel
// over P/4 + P/16 + ... (about P/3 pixels), 4.3 P reads; step 3, 9 per
// pixel over levels 0 to 4, about 3 P; step 4, 1 full-size read and 9 of
// B_0, 10 P; 17 P reads in all, most of them of half floats at reduced
// size, and 12 dispatches with 11 barriers. At 3456 x 2234 (P = 7.7 M) some
// 130 M reads; expected under a millisecond on the M3 Max, measured at
// implementation.
class ToneMapPass {
public:
    // Throws Error if `settings` are out of range (core/frame/tone_map.h) or
    // the device cannot make the pipelines or the settings' buffer.
    ToneMapPass(const Device& device, const Library& library, Submission& submission,
                const frame::ToneMap& settings);

    // Records the pass. Throws Error if `resources` has no radiance image or
    // no bloom pyramid.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> down_first_;  // step 2, from the exposed radiance
    NS::SharedPtr<MTL::ComputePipelineState> down_;        // step 2, from a level
    NS::SharedPtr<MTL::ComputePipelineState> up_;          // step 3
    NS::SharedPtr<MTL::ComputePipelineState> finish_;      // steps 4 to 6
    NS::SharedPtr<MTL::Buffer> settings_;
};

}  // namespace serenity::metal
