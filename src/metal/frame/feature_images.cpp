#include "metal/frame/feature_images.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "metal/device/error.h"
#include "metal/device/readback.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

constexpr std::size_t rgba = 4;  // channels a texel of the RGBA16Float images

// An image of `size` from `texels`, `channels` values a texel of type T,
// rows from the top: `pick(texel, c)` gives channel c of the three written.
template <typename T, typename Pick>
contracts::LinearImage widened(frame::Extent size, std::span<const T> texels,
                               std::size_t channels, Pick pick) {
  contracts::LinearImage image = contracts::make_linear_image(size);
  const std::size_t pixels = std::size_t{size.width} * size.height;
  for (std::size_t p = 0; p < pixels; ++p) {
    const std::span<const T> texel = texels.subspan(p * channels, channels);
    for (std::size_t c = 0; c < 3; ++c) {
      image.rgb[p * 3 + c] = pick(texel, c);
    }
  }
  return image;
}

}  // namespace

FeatureImages::FeatureImages(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

void FeatureImages::prepare(frame::Extent size) {
  check_texture_size(size, "feature images");
  if (size == size_) {
    return;
  }
  // Frames in flight may still read the old images (frame_images.h,
  // FrameImages::prepare, the same reasons).
  size_ = {};
  if (albedo_) {
    submission_.drain();
  }
  const auto pool = scoped_pool();
  const auto make = [&](ResidentTexture& image, MTL::PixelFormat format,
                        std::string_view what) {
    image.reset();
    image =
        make_resident_texture(device_.get(), submission_, format, size, what);
  };
  make(albedo_, MTL::PixelFormatRGBA16Float, "albedo image");
  make(normal_, MTL::PixelFormatRGBA16Float, "normal image");
  make(depth_, MTL::PixelFormatR32Float, "depth image");
  make(motion_, MTL::PixelFormatRGBA16Float, "motion image");
  size_ = size;
}

FeatureImages::Read FeatureImages::read() const {
  if (!albedo_.texture) {
    throw MetalError(
        "FeatureImages::read: no frame has made the feature images");
  }
  const std::size_t pixels = std::size_t{size_.width} * size_.height;
  const std::size_t half_row = std::size_t{size_.width} * rgba * sizeof(Half);
  const std::size_t float_row = std::size_t{size_.width} * sizeof(float);
  const std::size_t half_bytes = half_row * size_.height;
  // Albedo, normal and motion, then depth, in one shared buffer.
  const std::array<std::size_t, 4> offsets = {0, half_bytes, 2 * half_bytes,
                                              3 * half_bytes};
  const std::size_t total = 3 * half_bytes + float_row * size_.height;
  const std::array<ImageCopy, 4> copies = {
      ImageCopy{albedo_.texture.get(), offsets[0], half_row},
      ImageCopy{normal_.texture.get(), offsets[1], half_row},
      ImageCopy{motion_.texture.get(), offsets[2], half_row},
      ImageCopy{depth_.texture.get(), offsets[3], float_row}};
  const auto buffer = read_back(device_.get(), submission_, size_, total,
                                copies, "FeatureImages::read");

  // Copied out of the buffer as the values they are (ES.48: no cast of its
  // bytes to another type).
  const auto* const bytes = static_cast<const std::byte*>(buffer->contents());
  const auto halves = [&](std::size_t offset) {
    std::vector<Half> texels(pixels * rgba);
    std::memcpy(texels.data(), bytes + offset, half_bytes);
    return widened(size_, std::span<const Half>(texels), rgba,
                   [](std::span<const Half> texel, std::size_t c) {
                     return static_cast<float>(texel[c]);
                   });
  };
  std::vector<float> depths(pixels);
  std::memcpy(depths.data(), bytes + offsets[3], pixels * sizeof(float));
  return Read{
      .albedo = halves(offsets[0]),
      .normal = halves(offsets[1]),
      .depth = widened(
          size_, std::span<const float>(depths), 1,
          [](std::span<const float> texel, std::size_t) { return texel[0]; }),
      .motion = halves(offsets[2]),
  };
}

}  // namespace serenity::metal
