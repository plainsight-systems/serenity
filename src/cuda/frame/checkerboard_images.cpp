#include "cuda/frame/checkerboard_images.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

CheckerboardImages::CheckerboardImages(const Device& device,
                                       Submission& submission, Filled filled)
    : device_(device), submission_(submission), keeps_(filled) {}

bool CheckerboardImages::prepare(const frame::FrameInputs& inputs,
                                 frame::Extent size) {
  // Joined on a copy, kept once the images exist (E.4).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  try {
    held = joined.join(inputs, size).held;
  } catch (const frame::HistoryError& refused) {
    throw CudaError(std::string("the checkerboard fill's history: ") +
                    refused.what());
  }
  if (size == size_) {
    history_ = joined;
    return held > 0;
  }
  if (light_[0]) {
    submission_.finish();  // frames in flight may still read the old images
  }
  size_ = {};
  for (std::unique_ptr<PitchedImage>& image : light_) {
    image.reset();
    image = std::make_unique<PitchedImage>(device_, submission_, size);
  }
  filled_.reset();
  if (keeps_ == Filled::own) {
    filled_ = std::make_unique<PitchedImage>(device_, submission_, size);
  }
  size_ = size;
  history_ = joined;
  return held > 0;
}

}  // namespace serenity::cuda
