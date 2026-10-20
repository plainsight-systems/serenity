#include "cuda/frame/depth_of_field_images.h"

#include <cstddef>
#include <memory>

#include "core/passes/depth_of_field.h"

namespace serenity::cuda {

DepthOfFieldImages::DepthOfFieldImages(Submission& submission)
    : submission_(submission) {}

void DepthOfFieldImages::prepare(frame::Extent size) {
  if (size == size_) {
    return;
  }
  const frame::Extent half = passes::depth_of_field_half(size);
  // Frames in flight may still read the old images (frame_images.h, the
  // same reasons).
  size_ = {};
  if (images_[0]) {
    submission_.finish();
  }
  for (std::unique_ptr<LevelImage>& image : images_) {
    image.reset();
  }
  tiles_ = DeviceBuffer();
  dilated_ = DeviceBuffer();
  for (std::unique_ptr<LevelImage>& image : images_) {
    image = std::make_unique<LevelImage>(half);
  }
  const frame::Extent grid = passes::depth_of_field_tiles(size);
  const std::size_t tiles = std::size_t{grid.width} * grid.height;
  tiles_ = DeviceBuffer(tiles * sizeof(float));
  dilated_ = DeviceBuffer(tiles * sizeof(float));
  size_ = size;
}

}  // namespace serenity::cuda
