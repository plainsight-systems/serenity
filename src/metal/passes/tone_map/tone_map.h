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
// The frame's radiance as the look wants it shown: exposed, its brightest
// light spread into glare, rolled off toward white instead of clipped, and
// encoded for the display. The pass the window and movies end in
// (graphs/path.toml); its numbers are the graph's (core/frame/tone_map.h).
//
// Why it is needed: a firefly bright enough to light the scene is far
// brighter than the display shows, at its base glow and at its flash alike,
// so its body clips to one brightness either way (scenes/
// brass_sphere_flight.toml). An eye or a camera shows such a light as glare,
// a halo whose size and strength grow with its brightness; bloom is that
// glare, and with it a flash shows on the firefly itself.
//
// The algorithm, whose steps the code carries by number, over the radiance
// image L (metal/frame/frame_images.h) and the bloom pyramid B_0 .. B_5,
// each level half the one before (1/2 to 1/64 of the frame):
//
//   Step 1  Exposure: E = 2^exposure L, wherever L is read below.
//   Step 2  Down: B_0 is E filtered to half size, B_k is B_(k-1) filtered to
//           half size again, k = 1 .. 5, each by Jimenez's 13-tap filter
//           (SIGGRAPH 2014, "Next Generation Post Processing in Call of
//           Duty: Advanced Warfare"): five overlapping 2 x 2 box averages,
//           the middle one weighted 1/2 and the four corner ones 1/8 each,
//           taken with 13 bilinear reads. Edges clamp, so a uniform image
//           stays uniform at every level. Not the Karis average Jimenez
//           applies to the first level: it weights a pixel by 1 / (1 + its
//           luminance) to keep isolated over-bright pixels from blooming,
//           and the isolated over-bright pixels here are the fireflies,
//           whose glare is the point. The path tracer's own bright noise
//           blooms with them, until a denoiser takes the noise out first.
//   Step 3  Up: for k = 4 down to 0, B_k += tent(B_(k+1)), the 3 x 3 tent
//           filter (weights 1, 2, 1 by 1, 2, 1, over 16) of the level below,
//           read bilinearly at B_k's size. B_0 is then the sum of six blurs
//           of E, from narrow to wide: glare that is bright near the light
//           and wide around it, as a lens's is.
//   Step 4  Composite: C = (1 - bloom) E + bloom tent(B_0) / 6, at the
//           frame's size. Each blur keeps E's mean, so their sum's sixth
//           does, and C is a mean of E and it: bloom moves light and makes
//           none.
//   Step 5  Roll-off: Khronos' PBR Neutral tone mapper (KhronosGroup/
//           ToneMapping, PBR_Neutral, 2024), from its published equations,
//           with F90 = 0.04: K_s = 0.8 - F90, K_d = 0.15; x the smallest
//           channel of C, C -= (x < 0.08 ? x - 6.25 x^2 : 0.04); p its
//           largest; if p >= K_s, with d = 1 - K_s, p_n = 1 - d^2 / (p + d -
//           K_s), C *= p_n / p, and C mixed toward (p_n, p_n, p_n) by
//           1 - 1 / (K_d (p - p_n) + 1). Below K_s colors pass as they are;
//           above, they roll off toward 1 and toward white, so a firefly's
//           core goes white-hot while its glare keeps its yellow. Chosen over
//           ACES and AgX because it keeps hues where they are: AgX moves
//           brass's hue (a Blender user measured 52 degrees to 46), and the
//           brass and the fireflies are the scene's colors.
//   Step 6  Encode: sRGB's transfer function (passes/display.metal.h), into
//           the target.
//
// Steps 2 and 3 are 11 dispatches, one per level, and steps 4 to 6 one at
// the frame's size; each reads what the one before wrote, so a barrier from
// dispatch to dispatch sits between each two (GPU.7). The radiance image
// was written by an earlier pass; the renderer records that barrier
// (metal/frame/renderer.h).
//
// The settings reach the shader in a 16-byte buffer of the pass's own, made
// at construction and never written again: the graph's, the same every
// frame.
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
