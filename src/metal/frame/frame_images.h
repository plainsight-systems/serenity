#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/passes/tone_map.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the images between passes: contract 6).
//
// The images a frame's passes hand one another within the frame, as the
// schedule says they do (core/frame/schedule.h):
//
//   - the radiance image: the frame's linear radiance, RGBA32Float at the
//     frame's size, written by the pass that computes light (preview,
//     path) and read by the pass that presents it (display, tone_map).
//     32-bit, not 16: the display pass must show exactly what the light
//     pass computed, as the passes did when they encoded the target
//     themselves, and a firefly's radiance of hundreds keeps its digits;
//   - the bloom pyramid, for the tone-map pass alone: passes::bloom_levels
//     levels (core/passes/tone_map.h), each max(1, ceil(previous / 2)) on
//     each axis from the frame's size, so any frame has every level;
//     RGBA16Float, which a blur of light needs no more than (half's 11 bits
//     are 3 decimal digits), the light kept within its range
//     (core/passes/tone_map.h, steps 1 to 3).
//
// Made when the frame's size is first known and remade when it changes,
// before the frame's submission begins (Renderer::prepare), draining the
// submission first, as the accumulated image is (accumulation.h): frames in
// flight may still read the old images. Private storage, the GPU's alone,
// made resident through the submission. Only what the schedule uses is
// made: a schedule with no light pass makes none; one without tone_map, no
// pyramid.
//
// One set, not one per frame in flight: the frames in flight share it, and
// what keeps one frame's passes from writing it while the frame before still
// reads it is a barrier the renderer records before the first pass of a
// frame that writes it, waiting for the queue's earlier dispatches
// (metal/frame/renderer.h). So the CPU still records a frame while the GPU
// runs the one before; what the barrier gives up is only the GPU starting a
// frame's first dispatch before the last frame's final one ends. A copy per
// frame in flight would keep that sliver of overlap at twice the memory,
// 290 MB at the display's size, and the overlap is no gain: two frames'
// passes running at once take longer than one after the other (measured,
// docs/research/2026-10-09-pass-costs.md).
//
// Made outside the frame loop, at a size, and kept until the size changes
// (GPU.9); nothing else lives only within a frame for them to share memory
// with yet. The radiance image is an intermediate the frame writes and reads
// back, which GDSA.16 says to stream through on-chip memory instead: the
// tone map cannot, since its bloom reads every pixel's neighbourhood as far
// as B_5's 64 pixels; the display pass could fold back into the light pass,
// as it was, and is kept apart so a graph has one presenting pass
// (passes/display/display.h), at the cost derived below; it has not been
// measured on its own (docs/research/2026-10-09-pass-costs.md).
//
// Cost: at 3456 x 2234 (P = 7.7 M pixels), the radiance image is 123 MB
// and the pyramid 21 MB. Per frame, the light pass writes the radiance
// image once, P texels; the display pass reads it once, P texels, 247 MB of
// traffic in all, some 0.6 ms of the M3 Max's 400 GB/s. The tone-map pass
// reads it twice, in two dispatches ten apart, too far for one read to
// leave it cached for the other: 13 filtered samples for each of B_0's P / 4
// texels, then P exact reads (passes/tone_map/tone_map.h); what that costs
// in memory traffic is the pass's measured time, not a count here.
//
// Throws Error if the frame's size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class FrameImages {
public:
    // Whether the schedule has the tone-map pass, which needs the pyramid.
    // A type, not a bool, so the call says which (I.4).
    enum class Bloom { none, pyramid };

    // The radiance image always (a schedule with no light pass makes no
    // FrameImages), and the pyramid as `bloom` says.
    FrameImages(const Device& device, Submission& submission, Bloom bloom);

    FrameImages(const FrameImages&) = delete;
    FrameImages& operator=(const FrameImages&) = delete;
    FrameImages(FrameImages&&) = delete;
    FrameImages& operator=(FrameImages&&) = delete;
    ~FrameImages() = default;

    // Ready for a frame of `size`: made, or remade, if not made at it.
    void prepare(frame::Extent size);

    // Null when the schedule does not use it, or before the first prepare().
    // bloom() throws Error for a level past passes::bloom_levels.
    MTL::Texture* radiance() const noexcept { return radiance_.texture.get(); }
    MTL::Texture* bloom(std::uint32_t level) const;

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    Bloom bloom_;
    frame::Extent size_;
    // Each image with its residency, released after the GPU is done with it
    // (submission.h, Lifetime).
    struct Image {
        NS::SharedPtr<MTL::Texture> texture;
        Resident resident;
    };
    Image radiance_;
    std::array<Image, passes::bloom_levels> pyramid_;
};

}  // namespace serenity::metal
