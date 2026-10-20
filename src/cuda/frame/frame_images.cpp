#include "cuda/frame/frame_images.h"

#include <string>

#include "core/passes/bloom.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

FrameImages::FrameImages(Submission& submission) : submission_(submission) {}

void FrameImages::prepare(frame::Extent size) {
  if (size_ && *size_ == size) {
    return;
  }
  if (size_) {
    submission_.finish();  // frames in flight may still read the old levels
  }
  size_.reset();
  frame::Extent at = size;
  for (std::unique_ptr<LevelImage>& level : pyramid_) {
    at = passes::bloom_level(at);
    level = std::make_unique<LevelImage>(at);
  }
  size_ = size;
}

const LevelImage& FrameImages::level(std::uint32_t level) const {
  if (!size_ || level >= pyramid_.size()) {
    throw CudaError("FrameImages: no bloom level " + std::to_string(level));
  }
  return *pyramid_.at(level);
}

}  // namespace serenity::cuda
