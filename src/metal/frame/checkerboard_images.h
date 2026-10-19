#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstdint>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
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
// Whether the other image holds this frame's frame before is the core's
// reuse history (core/frame/history.h, ReuseHistory), kept here, as the
// denoiser keeps its own (denoise_images.h): prepare() joins the frame to
// it, so every graph whose light is the network's input asks the same rule
// of the same inputs, the denoised graph and the data set's input graph
// alike (logical-overview.md, principle 13), and the fill is told the same.
//
// A graph without the denoise pass hands its filled light to the presenting
// pass and the headless renderer instead of the network: for it, `filled`,
// one more RGBA32Float image at the light size, which the fill writes as it
// writes a denoised graph's hand-off (renderer.h, "Checkerboard
// rendering"); one, not one per frame slot, since such a graph's frames
// run on one queue, each after the frame before by its queue barrier.
//
// Made when the light's size is first known and remade when it changes,
// before the frame's submission begins, draining the submission first, as
// the feature images are (feature_images.h, the same reasons: GPU.9, MEM.9);
// private, the GPU's alone, written whole by the fill before any frame
// reads them as history. Memory, at P light pixels: 2 x 16 P bytes, 62 MB
// at 1728 x 1117; with `filled`, 16 P bytes more.
//
// Throws MetalError if the size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class CheckerboardImages {
 public:
  // Whether the graph keeps `filled` (above).
  enum class Filled : std::uint8_t { none, kept };

  CheckerboardImages(const Device& device, Submission& submission,
                     Filled filled);

  CheckerboardImages(const CheckerboardImages&) = delete;
  CheckerboardImages& operator=(const CheckerboardImages&) = delete;
  CheckerboardImages(CheckerboardImages&&) = delete;
  CheckerboardImages& operator=(CheckerboardImages&&) = delete;
  ~CheckerboardImages() = default;

  // Ready for frame `inputs` at the light's `size`: made, or remade, if
  // not made at it; and whether the image before(inputs.index) holds its
  // frame before (above). Throws MetalError if the core's history rule
  // refuses the frame.
  bool prepare(const frame::FrameInputs& inputs, frame::Extent size);

  // The image frame `index` writes, and the one it reads, its frame
  // before's (above). Null before the first prepare().
  MTL::Texture* written(std::uint64_t index) const noexcept {
    return light_[index % 2u].texture.get();
  }
  MTL::Texture* before(std::uint64_t index) const noexcept {
    return light_[(index + 1u) % 2u].texture.get();
  }

  // The filled light of a graph without the denoise pass (above); null in
  // a denoised graph and before the first prepare().
  MTL::Texture* filled() const noexcept { return filled_.texture.get(); }

 private:
  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  Filled kept_;
  frame::Extent size_;
  frame::ReuseHistory history_;
  std::array<ResidentTexture, 2> light_;
  ResidentTexture filled_;
};

}  // namespace serenity::metal
