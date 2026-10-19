#include "metal/frame/checkerboard_images.h"

#include <cstdint>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

CheckerboardImages::CheckerboardImages(const Device& device,
                                       Submission& submission, Filled filled)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      kept_(filled) {}

bool CheckerboardImages::prepare(const frame::FrameInputs& inputs,
                                 frame::Extent size) {
  check_texture_size(size, "checkerboard fill's history");
  // Joined on a copy, kept once the images exist (E.4; denoise_images.cpp,
  // the same rule).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  try {
    held = joined.join(inputs, size).held;
  } catch (const frame::HistoryError& refused) {
    throw MetalError(std::string("the checkerboard fill's history: ") +
                     refused.what());
  }
  if (size == size_) {
    history_ = joined;
    return held > 0;
  }
  // Frames in flight may still read the old images (feature_images.h,
  // FeatureImages::prepare, the same reasons).
  size_ = {};
  if (light_[0]) {
    submission_.drain();
  }
  const auto pool = scoped_pool();
  for (ResidentTexture& image : light_) {
    image.reset();
    image = make_resident_texture(device_.get(), submission_,
                                  MTL::PixelFormatRGBA32Float, size,
                                  "checkerboard fill's light and depth");
  }
  filled_.reset();
  if (kept_ == Filled::kept) {
    filled_ = make_resident_texture(device_.get(), submission_,
                                    MTL::PixelFormatRGBA32Float, size,
                                    "checkerboard fill's light");
  }
  size_ = size;
  history_ = joined;
  return held > 0;
}

}  // namespace serenity::metal
