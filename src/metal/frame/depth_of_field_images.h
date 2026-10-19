#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the depth of field's images between its dispatches:
// contract 6).
//
// What the depth-of-field pass (passes/depth_of_field/depth_of_field.h)
// keeps between its own dispatches within a frame, at the gather's grid,
// half the output's size each way, rounded up (contract 21):
//
//   presort  RGBA16Float: a half pixel's color, and its signed circle of
//            confusion, in half pixels, in alpha (an edge of 2^-11 of a
//            radius of 1, ample);
//   far      RGBA16Float: the gather's far field (rgb; alpha unused);
//   near     RGBA16Float: its near field, color and alpha;
//   far and near filtered: the median's output, the composite's input;
//
// and per tile of the grid two floats, the tile's largest near radius and
// its dilation (buffers, a float a tile).
//
// One set, not one per frame in flight: only the image half's queue reads
// and writes them, every dispatch of a frame writing what it reads after a
// barrier, and the image half's queue barrier keeps frame N + 1's from
// starting before frame N's have finished (renderer.h, "Two halves";
// GPU.8). Nothing in them outlives the frame.
//
// Made when the output's size is first known and remade when it changes,
// before the frame's submission begins, draining the submission first, as
// the other images between passes are (frame_images.h; GPU.9, MEM.9):
// private, the GPU's alone, every texel written by its dispatch before
// any reads it. Memory, at P output pixels: 5 x 8 x P / 4 bytes and the
// tiles', 10 P bytes, 77 MB at 3456 x 2234.
//
// Throws MetalError if the half size is not one Metal makes an image of,
// or the device cannot make one.
class DepthOfFieldImages {
 public:
  DepthOfFieldImages(const Device& device, Submission& submission);

  DepthOfFieldImages(const DepthOfFieldImages&) = delete;
  DepthOfFieldImages& operator=(const DepthOfFieldImages&) = delete;
  DepthOfFieldImages(DepthOfFieldImages&&) = delete;
  DepthOfFieldImages& operator=(DepthOfFieldImages&&) = delete;
  ~DepthOfFieldImages() = default;

  // Ready for a frame of the output's `size`: made, or remade, if not made
  // at it.
  void prepare(frame::Extent size);

  // Null, and 0, before the first prepare().
  MTL::Texture* presort() const noexcept { return images_[0].texture.get(); }
  MTL::Texture* far() const noexcept { return images_[1].texture.get(); }
  MTL::Texture* near() const noexcept { return images_[2].texture.get(); }
  MTL::Texture* far_filtered() const noexcept {
    return images_[3].texture.get();
  }
  MTL::Texture* near_filtered() const noexcept {
    return images_[4].texture.get();
  }
  MTL::GPUAddress tiles() const noexcept { return tiles_.address(); }
  MTL::GPUAddress dilated() const noexcept { return dilated_.address(); }

 private:
  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  frame::Extent size_;
  std::array<ResidentTexture, 5> images_;
  ResidentBuffer tiles_;
  ResidentBuffer dilated_;
};

}  // namespace serenity::metal
