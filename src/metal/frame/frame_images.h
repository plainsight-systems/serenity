#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
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
//   - the bloom pyramid, for the tone-map pass alone: six levels, each half
//     the one before in each direction (1/2 to 1/64 of the frame),
//     RGBA16Float, which a blur of light needs no more than (half's 11 bits
//     are 3 decimal digits, and its range reaches 65504).
//
// Made when the frame's size is first known and remade when it changes,
// before the frame's submission begins (Renderer::prepare), draining the
// submission first, as the accumulated image is (accumulation.h): frames in
// flight may still read the old images. Private storage, the GPU's alone,
// made resident through the submission. Only what the schedule uses is
// made: a schedule with no light pass makes none; one without tone_map, no
// pyramid.
//
// Not per frame in flight: a frame writes its images before it reads them
// and reads nothing of the last frame's, and the queue's frames run their
// dispatches in order (the path pass's barrier, path.h), so one set serves
// every frame.
//
// Cost: at 3456 x 2234, the radiance image is 123 MB and the pyramid 21 MB.
// Per frame: the radiance image written once and read once, 247 MB of
// traffic, some 0.6 ms of the M3 Max's 400 GB/s.
//
// Throws Error if the device cannot make an image.
class FrameImages {
public:
    static constexpr std::uint32_t bloom_levels = 6;

    // `radiance` and `pyramid`: which the schedule uses.
    FrameImages(const Device& device, Submission& submission, bool radiance, bool pyramid);

    FrameImages(const FrameImages&) = delete;
    FrameImages& operator=(const FrameImages&) = delete;
    FrameImages(FrameImages&&) = delete;
    FrameImages& operator=(FrameImages&&) = delete;
    ~FrameImages() = default;

    // Ready for a frame of `size`: made, or remade, if not made at it.
    void prepare(frame::Extent size);

    // Null when the schedule does not use it, or before the first prepare().
    MTL::Texture* radiance() const { return radiance_.get(); }
    MTL::Texture* bloom(std::uint32_t level) const { return pyramid_.at(level).get(); }

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    bool wants_radiance_;
    bool wants_pyramid_;
    frame::Extent size_;
    NS::SharedPtr<MTL::Texture> radiance_;
    std::array<NS::SharedPtr<MTL::Texture>, bloom_levels> pyramid_;
};

}  // namespace serenity::metal
