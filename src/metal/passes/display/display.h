#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (display).
//
// The frame's radiance as it is, for the display: per pixel, the radiance
// image's color (metal/frame/frame_images.h), a color brighter than the
// display shows scaled down by its largest channel, keeping its hue, then
// sRGB's transfer function (display.metal.h), into the target. What
// the preview and path passes did to their own output before the radiance
// image came between them and the target, moved here unchanged, so a frame
// graph of a light pass and this one shows, byte for byte, what that light
// pass alone showed before: the tests' graphs (tests/gpu) and the
// diagnostic ones (graphs/preview.toml).
//
// No exposure, no glare, no roll-off: those are tone_map's
// (passes/tone_map/tone_map.h), for the look; this is for seeing the
// numbers.
//
// The radiance image was written by an earlier pass in the frame; the
// renderer records the barrier between them (metal/frame/renderer.h).
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2); 16
// bytes read and 4 written a pixel, and a few flops.
class DisplayPass {
public:
    DisplayPass(const Device& device, const Library& library);

    // Records the pass. Throws Error if `resources` has no radiance image.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
