#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (test pattern).
//
// A diagnostic pass, kept for good: it shows that a frame reaches the screen,
// or a file, with nothing of the renderer in the way. When the image is wrong,
// this pass tells the presentation path apart from the rendering.
//
// Each pixel (x, y) of a W x H image is written as
//
//   red   = x / (W - 1)            across
//   green = y / (H - 1)            down
//   blue  = 0.5 + 0.5 sin(2 pi t / 4 s)   pulsing once every four seconds
//
// with alpha 1, from the frame's constants alone (contracts/frame_constants.h).
// A pure function of the pixel and the constants (F.8), so a test checks the
// pixels against the formula, within the rounding of 8-bit storage and of the
// GPU's sin.
//
// The kernel is test_pattern.metal, beside this file: one responsibility in
// two languages, kept as a pair (change-axes.md).
//
// Cost: one thread per pixel, each writing 4 bytes, in threadgroups of the
// pipeline's execution width by as many rows as the threadgroup limit allows.
// Adjacent threads write adjacent pixels of a row (GPU.2). At 3456 x 2234 it
// writes 30.9 MB a frame and reads nothing but the frame's 16 bytes of
// constants.
class TestPatternPass {
public:
    // Builds the pipeline for test_pattern from `library`. Throws Error if it
    // cannot.
    TestPatternPass(const Device& device, const Library& library);

    // Records the pass into `encoder`: binds the frame's constants and its
    // target through the resources' argument table, and dispatches one thread
    // per pixel. It reads no scene.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
