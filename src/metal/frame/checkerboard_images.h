#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the checkerboard fill's history: contract 6).
//
// What the fill (passes/checkerboard_fill/checkerboard_fill.h) keeps from
// one frame for the next: the light it handed the denoiser, every pixel of
// it, and the depth each pixel saw, which the next frame's fill reads to
// take an untraced pixel's light from where its surface was, where it is
// the same surface: one RGBA32Float image, rgb the frame's handed-off
// light, a the pixel's view depth (contract 17's depth, 0 for the sky).
//
// Two, by the frame's index's parity, as the denoiser keeps its
// output (denoise_images.h, history): frame n writes image n mod 2 and reads
// image (n - 1) mod 2. Not by the frame slot, as the hand-off is: a frame
// slot follows the submissions, and a readback between two frames (the
// headless renderer's PFM) is one, so two frames in a row may share a slot
// and the second would find the first's images already overwritten; the
// parity follows the frames themselves, so every frame reads its frame
// before however it was read back (logical-overview.md, principle 1: the
// same image in the window and headless). Only the light half's queue
// touches them, in frame order, so frame n + 1's write of the image frame
// n - 1 wrote waits for frame n's read of it by that queue's barrier
// (renderer.h, "Checkerboard rendering"; GPU.8); the image half reads the
// hand-off, never these (CP.3).
//
// Made when the light's size is first known and remade when it changes,
// before the frame's submission begins, draining the submission first, as
// the feature images are (feature_images.h, the same reasons: GPU.9, MEM.9);
// private, the GPU's alone, written whole by the fill before any frame
// reads them as history. Memory, at P light pixels: 2 x 16 P bytes, 62 MB
// at 1728 x 1117.
//
// Throws MetalError if the size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class CheckerboardImages {
public:
    CheckerboardImages(const Device& device, Submission& submission);

    CheckerboardImages(const CheckerboardImages&) = delete;
    CheckerboardImages& operator=(const CheckerboardImages&) = delete;
    CheckerboardImages(CheckerboardImages&&) = delete;
    CheckerboardImages& operator=(CheckerboardImages&&) = delete;
    ~CheckerboardImages() = default;

    // Ready for a frame of the light's `size`: made, or remade, if not made
    // at it.
    void prepare(frame::Extent size);

    // The image frame `index` writes, and the one it reads, its frame
    // before's (above). Null before the first prepare().
    MTL::Texture* written(std::uint64_t index) const noexcept { return light_[index % 2u].texture.get(); }
    MTL::Texture* before(std::uint64_t index) const noexcept { return light_[(index + 1u) % 2u].texture.get(); }

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    frame::Extent size_;
    std::array<ResidentTexture, 2> light_;
};

}  // namespace serenity::metal
