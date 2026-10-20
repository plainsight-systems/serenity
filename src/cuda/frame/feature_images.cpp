#include "cuda/frame/feature_images.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/contracts/linear_image.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

constexpr std::size_t rgba = 4;  // halves a pixel of the four-channel images

// A half float on the host: _Float16, a compiler extension (P.2), as
// Metal's host widens its own (metal/device/support.h).
using Half = _Float16;

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

FeatureImages::FeatureImages(Submission& submission)
    : submission_(submission) {}

void FeatureImages::prepare(frame::Extent size) {
  if (size == size_) {
    return;
  }
  if (albedo_.address() != 0) {
    submission_.finish();  // frames in flight may still read the old images
  }
  size_ = {};
  const std::size_t pixels = std::size_t{size.width} * size.height;
  albedo_ = DeviceBuffer(pixels * rgba * sizeof(std::uint16_t));
  normal_ = DeviceBuffer(pixels * rgba * sizeof(std::uint16_t));
  motion_ = DeviceBuffer(pixels * rgba * sizeof(std::uint16_t));
  depth_ = DeviceBuffer(pixels * sizeof(float));
  size_ = size;
}

FeatureImages::Read FeatureImages::read() const {
  if (albedo_.address() == 0) {
    throw CudaError(
        "FeatureImages::read: no frame has made the feature images");
  }
  submission_.finish();
  const std::size_t pixels = std::size_t{size_.width} * size_.height;
  const auto halves = [&](const DeviceBuffer& image) {
    std::vector<std::uint16_t> bits(pixels * rgba);
    check(cuMemcpyDtoH(bits.data(), image.address(),
                       bits.size() * sizeof(std::uint16_t)),
          "FeatureImages::read: cuMemcpyDtoH");
    return widened(size_, std::span<const std::uint16_t>(bits), rgba,
                   [](std::span<const std::uint16_t> texel, std::size_t c) {
                     return static_cast<float>(std::bit_cast<Half>(texel[c]));
                   });
  };
  std::vector<float> depths(pixels);
  check(cuMemcpyDtoH(depths.data(), depth_.address(), pixels * sizeof(float)),
        "FeatureImages::read: cuMemcpyDtoH");
  return Read{
      .albedo = halves(albedo_),
      .normal = halves(normal_),
      .depth = widened(
          size_, std::span<const float>(depths), 1,
          [](std::span<const float> texel, std::size_t) { return texel[0]; }),
      .motion = halves(motion_),
  };
}

}  // namespace serenity::cuda
