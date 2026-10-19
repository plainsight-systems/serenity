#include "metal/frame/depth_of_field_images.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/passes/depth_of_field.h"
#include "metal/device/support.h"

namespace serenity::metal {

DepthOfFieldImages::DepthOfFieldImages(const Device& device,
                                       Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

void DepthOfFieldImages::prepare(frame::Extent size) {
  if (size == size_) {
    return;
  }
  const frame::Extent half = passes::depth_of_field_half(size);
  check_texture_size(half, "depth of field's images");
  // Frames in flight may still read the old images (frame_images.h, the
  // same reasons).
  size_ = {};
  if (images_[0]) {
    submission_.drain();
  }
  for (ResidentTexture& image : images_) {
    image.reset();
  }
  tiles_.reset();
  dilated_.reset();
  const auto pool = scoped_pool();
  constexpr std::array<std::string_view, image_count> what = {
      "depth of field's presort", "depth of field's far field",
      "depth of field's near field", "depth of field's filtered far field",
      "depth of field's filtered near field"};
  for (std::size_t i = 0; i < images_.size(); ++i) {
    images_.at(i) =
        make_resident_texture(device_.get(), submission_,
                              MTL::PixelFormatRGBA16Float, half, what.at(i));
  }
  const frame::Extent grid = passes::depth_of_field_tiles(size);
  const std::size_t tiles = std::size_t{grid.width} * grid.height;
  tiles_ = make_resident_buffer(
      device_.get(), submission_, tiles * sizeof(float),
      MTL::ResourceStorageModePrivate, "depth of field's tiles");
  dilated_ = make_resident_buffer(
      device_.get(), submission_, tiles * sizeof(float),
      MTL::ResourceStorageModePrivate, "depth of field's dilated tiles");
  size_ = size;
}

}  // namespace serenity::metal
