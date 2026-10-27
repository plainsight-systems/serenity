#include "metal/frame/denoise_images.h"

#include <array>
#include <cstdint>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

DenoiseImages::DenoiseImages(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

void DenoiseImages::make(Buffer& buffer, std::size_t bytes,
                         std::string_view what) {
  buffer = make_resident_buffer(
      device_.get(), submission_, bytes, MTL::ResourceStorageModeShared,
      "the denoiser's " + std::string(what), Zeroed::yes);
}

bool DenoiseImages::prepare(const frame::FrameInputs& inputs,
                            frame::Extent light, frame::Extent grid) {
  check_texture_size(light, "the denoiser's buffers");
  if (grid.width < light.width || grid.height < light.height) {
    throw MetalError("the denoiser's grid " + frame::to_string(grid) +
                     " does not hold its light, " + frame::to_string(light));
  }
  // Joined on a copy, kept once the buffers exist (E.4; frame_images.cpp,
  // PathReuseImages::prepare).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  try {
    held = joined.join(inputs, light).held;
  } catch (const frame::HistoryError& refused) {
    throw MetalError(std::string("the denoiser's history: ") + refused.what());
  }
  if (light == light_ && grid == padded_) {
    history_ = joined;
    index_ = inputs.index;
    return held > 0;
  }
  // A new size starts the history over (the rule refuses one that does
  // not); frames in flight may still read the old buffers.
  if (demod_) {
    submission_.drain();
  }
  light_ = {};
  const auto pool = scoped_pool();
  padded_ = grid;
  allocate(light);
  light_ = light;
  history_ = joined;
  index_ = inputs.index;
  // Buffers just made hold no last frame, whatever the history holds (a
  // new grid at the same light).
  return false;
}

void DenoiseImages::allocate(frame::Extent light) {
  const std::size_t input_pixels = std::size_t{padded_.width} * padded_.height;
  // The demodulated light and the histories: a float4 a pixel.
  make(demod_, input_pixels * 4 * sizeof(float), "demodulated light");
  for (Buffer& h : history_buffers_) {
    make(h, input_pixels * contracts::denoise_subpixels * 4 * sizeof(float),
         "history");
  }
  make(histogram_, contracts::median_bins * sizeof(std::uint32_t),
       "depth histogram");
  make(median_, sizeof(contracts::MedianState), "depth median");
  // The hand-off, at the light's own size, the accumulated image's format.
  for (ResidentTexture& image : light_radiance_) {
    image.reset();
    image = make_resident_texture(device_.get(), submission_,
                                  MTL::PixelFormatRGBA32Float, light,
                                  "light radiance hand-off");
  }
}

MTL::Texture* DenoiseImages::light_radiance(std::uint32_t slot) const {
  if (slot >= light_radiance_.size()) {
    throw MetalError("DenoiseImages::light_radiance: slot " +
                     std::to_string(slot) + " is not one");
  }
  return light_radiance_.at(slot).texture.get();
}

MTL::GPUAddress DenoiseImages::demod() const noexcept {
  return demod_.address();
}
MTL::GPUAddress DenoiseImages::histogram() const noexcept {
  return histogram_.address();
}
MTL::GPUAddress DenoiseImages::median() const noexcept {
  return median_.address();
}
MTL::GPUAddress DenoiseImages::history() const noexcept {
  return history_buffers_.at((index_ + 1) % 2).address();
}
MTL::GPUAddress DenoiseImages::next_history() const noexcept {
  return history_buffers_.at(index_ % 2).address();
}

}  // namespace serenity::metal
