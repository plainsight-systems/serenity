#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <memory>

#include "core/frame/extent.h"
#include "cuda/device/buffer.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (the depth of field's images between its launches:
// contract 6).
//
// What the depth-of-field pass (cuda/passes/depth_of_field/
// depth_of_field.h) keeps between its own launches within a frame, Metal's
// DepthOfFieldImages (metal/frame/depth_of_field_images.h, which says what
// each holds) at the gather's grid, half the output's size each way,
// rounded up (contract 21, core/passes/depth_of_field.h):
//
//   presort  four half floats a pixel: a half pixel's color, and its
//            signed circle of confusion, in half pixels, in alpha;
//   far      the gather's far field (rgb; alpha unused);
//   near     its near field, color and alpha;
//   far and near filtered: the median's output, the composite's input;
//
// each a LevelImage (cuda/device/images.h): Metal's RGBA16Float, written
// and read exactly through its surface object, the filtered two also read
// bilinearly through its texture object by the composite, as the tone map
// reads its bloom levels. And per tile of the grid a float, the tile's
// largest near radius, and another, its dilation: device buffers, a float
// a tile, rows of tiles tightly packed.
//
// One set, not one per frame in flight, as Metal's: the frame's stream
// runs every launch of the pass in order and one frame's after the last's
// (GPU.8: the order is the stream's, no barrier), every launch writing what
// it reads before it is read, so nothing in them outlives the frame.
//
// Made when the output's size is first known and remade when it changes,
// before the frame's work is recorded, the frames in flight finished first
// (Submission::finish, which Metal's drain is), as the other images between
// passes are (frame_images.h; MEM.9: made outside the frame loop, GPU.9:
// one allocation an image, none a frame). Owned, released with this object
// (R.1). Memory, at P output pixels: 5 x 8 x P / 4 bytes and the tiles',
// 10 P bytes, 77 MB at 3456 x 2234, Metal's.
//
// Throws CudaError if an image or buffer cannot be made (the driver
// refuses an array of the half size, or memory runs out).
class DepthOfFieldImages {
 public:
  explicit DepthOfFieldImages(Submission& submission);

  DepthOfFieldImages(const DepthOfFieldImages&) = delete;
  DepthOfFieldImages& operator=(const DepthOfFieldImages&) = delete;
  DepthOfFieldImages(DepthOfFieldImages&&) = delete;
  DepthOfFieldImages& operator=(DepthOfFieldImages&&) = delete;
  ~DepthOfFieldImages() = default;

  // Ready for a frame of the output's `size`: made, or remade, if not made
  // at it.
  void prepare(frame::Extent size);

  // The output's size it was made for; empty before the first prepare().
  frame::Extent size() const noexcept { return size_; }

  // Null, and 0, before the first prepare().
  const LevelImage* presort() const noexcept { return images_[0].get(); }
  const LevelImage* far() const noexcept { return images_[1].get(); }
  const LevelImage* near() const noexcept { return images_[2].get(); }
  const LevelImage* far_filtered() const noexcept { return images_[3].get(); }
  const LevelImage* near_filtered() const noexcept { return images_[4].get(); }
  CUdeviceptr tiles() const noexcept { return tiles_.address(); }
  CUdeviceptr dilated() const noexcept { return dilated_.address(); }

 private:
  Submission& submission_;
  frame::Extent size_;
  // The presort, the far and near fields, and each filtered.
  static constexpr std::size_t image_count = 5;
  std::array<std::unique_ptr<LevelImage>, image_count> images_;
  DeviceBuffer tiles_;
  DeviceBuffer dilated_;
};

}  // namespace serenity::cuda
