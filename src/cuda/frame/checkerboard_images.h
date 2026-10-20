#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "cuda/device/device.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (the checkerboard fill's history: contract 6).
//
// What the fill (cuda/passes/checkerboard_fill/checkerboard_fill.h) keeps
// from one frame for the next, Metal's CheckerboardImages (metal/frame/
// checkerboard_images.h, which says why by parity and not by slot): two
// images of four floats a pixel, the frame's handed-off light and in w each
// pixel's view depth, frame n writing image n mod 2 and reading image
// (n - 1) mod 2; whether the other holds this frame's frame before, the
// core's reuse history (frame::ReuseHistory), kept here. And the filled
// light the data set's input graph shows and writes (`filled`): this
// backend builds no network yet, so every graph with the fill keeps it.
// Pitched (cuda/device/images.h), the filled image read filtered by the tone
// map as the accumulated image is.
//
// Made when the light's size is first known and remade when it changes,
// the frames in flight finished first. Throws CudaError if an image cannot
// be made or the core's history rule refuses a frame.
class CheckerboardImages {
 public:
  CheckerboardImages(const Device& device, Submission& submission);

  CheckerboardImages(const CheckerboardImages&) = delete;
  CheckerboardImages& operator=(const CheckerboardImages&) = delete;
  CheckerboardImages(CheckerboardImages&&) = delete;
  CheckerboardImages& operator=(CheckerboardImages&&) = delete;
  ~CheckerboardImages() = default;

  // Ready for frame `inputs` at `size`: made, or remade, if not made at it;
  // and whether before(inputs.index) holds its frame before.
  bool prepare(const frame::FrameInputs& inputs, frame::Extent size);

  // The image frame `index` writes, and the one it reads, its frame
  // before's. Null before the first prepare().
  const PitchedImage* written(std::uint64_t index) const noexcept {
    return light_.at(index % 2u).get();
  }
  const PitchedImage* before(std::uint64_t index) const noexcept {
    return light_.at((index + 1u) % 2u).get();
  }

  // The filled light; null before the first prepare().
  const PitchedImage* filled() const noexcept { return filled_.get(); }

 private:
  const Device& device_;
  Submission& submission_;
  frame::Extent size_;
  frame::ReuseHistory history_;
  std::array<std::unique_ptr<PitchedImage>, 2> light_;
  std::unique_ptr<PitchedImage> filled_;
};

}  // namespace serenity::cuda
